import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, EmailStr
from sklearn.ensemble import RandomForestRegressor
from starlette.middleware.cors import CORSMiddleware

from .db import get_db, close_client
from .auth import create_token, hash_password, verify_password, get_current_user

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

CORS_ORIGINS_RAW = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = ["*"] if CORS_ORIGINS_RAW.strip() == "*" else [
    o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()
]

app = FastAPI()
api_router = APIRouter(prefix="/api")


# -----------------------------
# Models
# -----------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: User


class Product(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    category: str
    current_price: float
    cost: float
    inventory: int
    demand_last_30d: int
    avg_competitor_price: float
    optimized_price: Optional[float] = None
    price_elasticity: Optional[float] = None
    updated_at: str


class OptimizationResult(BaseModel):
    product_id: str
    product_name: str
    current_price: float
    optimized_price: float
    predicted_revenue_lift: float
    predicted_demand_change: float
    recommendation: str


class DashboardMetrics(BaseModel):
    total_revenue: float
    revenue_lift: float
    avg_conversion_rate: float
    inventory_turnover: float
    total_products: int
    optimized_products: int


class ScenarioRequest(BaseModel):
    product_id: str
    test_price: float


class ScenarioResult(BaseModel):
    product_id: str
    test_price: float
    predicted_demand: int
    predicted_revenue: float
    revenue_change_pct: float


# -----------------------------
# Mock Data Init
# -----------------------------
async def initialize_mock_data() -> None:
    db = get_db()
    existing_products = await db.products.count_documents({})
    if existing_products > 0:
        return

    now = datetime.now(timezone.utc).isoformat()
    products_data = [
        {"id": str(uuid.uuid4()), "name": "Wireless Headphones", "category": "Electronics", "current_price": 79.99, "cost": 35.0, "inventory": 245, "demand_last_30d": 187, "avg_competitor_price": 82.50, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Smart Watch", "category": "Electronics", "current_price": 199.99, "cost": 95.0, "inventory": 156, "demand_last_30d": 143, "avg_competitor_price": 189.99, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Yoga Mat", "category": "Fitness", "current_price": 34.99, "cost": 12.0, "inventory": 423, "demand_last_30d": 312, "avg_competitor_price": 39.99, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Coffee Maker", "category": "Home", "current_price": 89.99, "cost": 42.0, "inventory": 178, "demand_last_30d": 156, "avg_competitor_price": 94.99, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Laptop Stand", "category": "Office", "current_price": 45.99, "cost": 18.0, "inventory": 289, "demand_last_30d": 234, "avg_competitor_price": 49.99, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Gaming Mouse", "category": "Electronics", "current_price": 59.99, "cost": 24.0, "inventory": 512, "demand_last_30d": 445, "avg_competitor_price": 54.99, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Water Bottle", "category": "Fitness", "current_price": 24.99, "cost": 8.0, "inventory": 678, "demand_last_30d": 589, "avg_competitor_price": 27.99, "updated_at": now},
        {"id": str(uuid.uuid4()), "name": "Desk Lamp", "category": "Office", "current_price": 39.99, "cost": 15.0, "inventory": 234, "demand_last_30d": 198, "avg_competitor_price": 44.99, "updated_at": now},
    ]
    await db.products.insert_many(products_data)

    demand_history = []
    products = await db.products.find({}, {"_id": 0}).to_list(None)

    for product in products:
        base_daily = max(1, int(product["demand_last_30d"] / 30))
        for day in range(90):
            date = (datetime.now(timezone.utc) - timedelta(days=90 - day)).date().isoformat()
            demand = max(0, int(base_daily + np.random.normal(0, base_daily * 0.3)))
            price_variation = float(product["current_price"]) * (1 + np.random.uniform(-0.15, 0.15))
            demand_history.append({
                "id": str(uuid.uuid4()),
                "date": date,
                "product_id": product["id"],
                "demand": int(demand),
                "price": round(price_variation, 2),
            })

    if demand_history:
        await db.demand_history.insert_many(demand_history)


@app.on_event("startup")
async def startup_event():
    await initialize_mock_data()


@app.on_event("shutdown")
async def shutdown_event():
    close_client()


@api_router.get("/health")
async def health():
    return {"status": "ok"}


# -----------------------------
# Auth
# -----------------------------
@api_router.post("/auth/register", response_model=AuthResponse)
async def register(user_data: UserRegister):
    db = get_db()
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password": hash_password(user_data.password),
        "created_at": created_at,
    }
    await db.users.insert_one(user_doc)

    token = create_token(user_id)
    return AuthResponse(
        token=token,
        user=User(id=user_id, email=user_data.email, name=user_data.name, created_at=created_at),
    )


@api_router.post("/auth/login", response_model=AuthResponse)
async def login(credentials: UserLogin):
    db = get_db()
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"])
    return AuthResponse(
        token=token,
        user=User(id=user["id"], email=user["email"], name=user["name"], created_at=user["created_at"]),
    )


# -----------------------------
# Products
# -----------------------------
@api_router.get("/products", response_model=List[Product])
async def get_products(current_user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.products.find({}, {"_id": 0}).to_list(None)


@api_router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    product = await db.products.find_one({"id": product_id}, {"_id": 0})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


# -----------------------------
# Optimization
# -----------------------------
@api_router.post("/optimize-prices", response_model=List[OptimizationResult])
async def optimize_prices(current_user: dict = Depends(get_current_user)):
    db = get_db()
    products = await db.products.find({}, {"_id": 0}).to_list(None)
    results: List[OptimizationResult] = []

    for product in products:
        demand_data = await db.demand_history.find({"product_id": product["id"]}, {"_id": 0}).to_list(None)
        if len(demand_data) < 10:
            continue

        df = pd.DataFrame(demand_data)
        df["price_change"] = df["price"].pct_change()
        df["demand_change"] = df["demand"].pct_change()
        df = df.dropna()

        if len(df) > 0:
            elasticity = (df["demand_change"] / df["price_change"]).replace([np.inf, -np.inf], 0).mean()
            elasticity = float(max(-3.0, min(-0.1, elasticity)))
        else:
            elasticity = -1.5

        X = df[["price"]].values
        y = df["demand"].values

        current_price = float(product["current_price"])
        competitor_price = float(product["avg_competitor_price"])
        inventory = int(product["inventory"])

        ml_optimal_price = current_price
        if len(X) >= 10:
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            model.fit(X, y)

            min_price = float(product["cost"]) * 1.2
            max_price = competitor_price * 1.3
            test_prices = np.linspace(min_price, max_price, 20).reshape(-1, 1)
            predicted_demands = model.predict(test_prices)

            revenues = test_prices.flatten() * predicted_demands
            ml_optimal_price = float(test_prices[int(np.argmax(revenues))][0])

        if inventory > 300:
            price_adjustment = -0.10
        elif inventory < 150:
            price_adjustment = 0.08
        else:
            price_adjustment = 0.0

        if current_price > competitor_price * 1.1:
            price_adjustment -= 0.05
        elif current_price < competitor_price * 0.9:
            price_adjustment += 0.05

        rule_optimal_price = current_price * (1 + price_adjustment)

        optimized_price = (0.7 * ml_optimal_price + 0.3 * rule_optimal_price)
        optimized_price = max(float(product["cost"]) * 1.15, min(competitor_price * 1.2, optimized_price))
        optimized_price = round(float(optimized_price), 2)

        predicted_demand_change = elasticity * ((optimized_price - current_price) / current_price) * 100.0
        current_revenue = current_price * int(product["demand_last_30d"])
        new_demand = int(product["demand_last_30d"]) * (1 + predicted_demand_change / 100.0)
        predicted_revenue = optimized_price * new_demand
        revenue_lift = ((predicted_revenue - current_revenue) / current_revenue) * 100.0 if current_revenue > 0 else 0.0

        if revenue_lift > 5:
            recommendation = "Implement immediately - High revenue potential"
        elif revenue_lift > 0:
            recommendation = "Consider implementing - Positive impact expected"
        else:
            recommendation = "Monitor - Minimal impact predicted"

        await db.products.update_one(
            {"id": product["id"]},
            {"$set": {
                "optimized_price": optimized_price,
                "price_elasticity": round(float(elasticity), 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
        )

        results.append(
            OptimizationResult(
                product_id=product["id"],
                product_name=product["name"],
                current_price=current_price,
                optimized_price=optimized_price,
                predicted_revenue_lift=round(float(revenue_lift), 2),
                predicted_demand_change=round(float(predicted_demand_change), 2),
                recommendation=recommendation,
            )
        )

    return results


# -----------------------------
# Dashboard Metrics
# -----------------------------
@api_router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(current_user: dict = Depends(get_current_user)):
    db = get_db()
    products = await db.products.find({}, {"_id": 0}).to_list(None)

    total_revenue = sum(float(p["current_price"]) * int(p["demand_last_30d"]) for p in products)
    optimized_revenue = sum(
        float(p.get("optimized_price") or p["current_price"]) * int(p["demand_last_30d"])
        for p in products
    )
    revenue_lift = ((optimized_revenue - total_revenue) / total_revenue * 100.0) if total_revenue > 0 else 0.0

    total_demand = sum(int(p["demand_last_30d"]) for p in products)
    total_inventory = sum(int(p["inventory"]) for p in products)

    avg_conversion = (total_demand / total_inventory * 100.0) if total_inventory > 0 else 0.0
    inventory_turnover = (total_demand / total_inventory) if total_inventory > 0 else 0.0

    optimized_count = sum(1 for p in products if p.get("optimized_price") is not None)

    return DashboardMetrics(
        total_revenue=round(float(total_revenue), 2),
        revenue_lift=round(float(revenue_lift), 2),
        avg_conversion_rate=round(float(avg_conversion), 2),
        inventory_turnover=round(float(inventory_turnover), 2),
        total_products=len(products),
        optimized_products=optimized_count,
    )


# -----------------------------
# Analytics
# -----------------------------
@api_router.get("/analytics/demand-trends/{product_id}")
async def get_demand_trends(product_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.demand_history.find({"product_id": product_id}, {"_id": 0}).sort("date", 1).to_list(None)


@api_router.get("/analytics/elasticity")
async def get_elasticity_analysis(current_user: dict = Depends(get_current_user)):
    db = get_db()
    products = await db.products.find({"price_elasticity": {"$exists": True}}, {"_id": 0}).to_list(None)
    return [{"product_id": p["id"], "product_name": p["name"], "elasticity": p.get("price_elasticity", 0)} for p in products]


# -----------------------------
# Scenarios
# -----------------------------
@api_router.post("/scenarios/test", response_model=ScenarioResult)
async def test_scenario(scenario: ScenarioRequest, current_user: dict = Depends(get_current_user)):
    db = get_db()
    product = await db.products.find_one({"id": scenario.product_id}, {"_id": 0})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    elasticity = float(product.get("price_elasticity", -1.5))
    current_price = float(product["current_price"])
    current_demand = int(product["demand_last_30d"])

    price_change_pct = (float(scenario.test_price) - current_price) / current_price
    demand_change_pct = elasticity * price_change_pct

    predicted_demand = int(current_demand * (1 + demand_change_pct))
    predicted_demand = max(0, predicted_demand)

    current_revenue = current_price * current_demand
    predicted_revenue = float(scenario.test_price) * predicted_demand
    revenue_change_pct = ((predicted_revenue - current_revenue) / current_revenue) * 100.0 if current_revenue > 0 else 0.0

    return ScenarioResult(
        product_id=scenario.product_id,
        test_price=float(scenario.test_price),
        predicted_demand=predicted_demand,
        predicted_revenue=round(float(predicted_revenue), 2),
        revenue_change_pct=round(float(revenue_change_pct), 2),
    )


# -----------------------------
# Competitors
# -----------------------------
@api_router.get("/competitors")
async def get_competitor_prices(current_user: dict = Depends(get_current_user)):
    db = get_db()
    products = await db.products.find({}, {"_id": 0}).to_list(None)

    out = []
    for p in products:
        our = float(p["current_price"])
        comp = float(p["avg_competitor_price"]) or 1.0
        out.append({
            "product_id": p["id"],
            "product_name": p["name"],
            "our_price": our,
            "competitor_avg": comp,
            "price_difference": round(our - comp, 2),
            "price_difference_pct": round(((our - comp) / comp) * 100.0, 2),
        })
    return out


# -----------------------------
# Middleware + Router
# -----------------------------
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)