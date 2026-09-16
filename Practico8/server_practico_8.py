from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


@mcp.tool()
async def get_product_by_name(name: str) -> list:
    """Search for a product by name. """
    products = [
        {
            "id": "P001",
            "name": "iPhone 17",
            "category": "Phones"
        },
        {
            "id": "P002",
            "name": "Teclado mecanico",
            "category": "Accessories"
        },
        {
            "id": "P003",
            "name": "MacBook Air",
            "category": "Computers"
        },
    ]

    results = [
        product
        for product in products
        if name.lower() in product["name"].lower()
    ]

    return results


@mcp.tool()
def get_product_price(product_id: str) -> dict:
    """
    Get the price of a product by its ID.
    """
    prices = {
        "P001": {
            "price": 999.99,
            "currency": "USD"
        },
        "P002": {
            "price": 899.99,
            "currency": "USD"
        },
        "P003": {
            "price": 1199.99,
            "currency": "USD"
        },
    }

    if product_id not in prices:
        return {"error": "Product not found"}

    return prices[product_id]


@mcp.tool()
def get_product_stock(product_id: str) -> dict:
    """
    Get the available stock for a product.
    """
    stock = {
        "P001": 10,
        "P002": 5,
        "P003": 3,
    }

    if product_id not in stock:
        return {"error": "Product not found"}

    return {
        "product_id": product_id,
        "available_stock": stock[product_id],
    }

@mcp.tool()
def list_product_categories() -> list:
    """
    List all available product categories.
    """
    return [
        "Phones",
        "Computers",
        "Tablets",
        "Accessories",
        "Audio",
    ]