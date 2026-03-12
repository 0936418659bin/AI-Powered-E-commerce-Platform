"use client";

import { ProductFilters } from "@/components/products/ProductFilters";
import { ProductGrid } from "@/components/products/ProductGrid";
import api from "@/lib/api";
import { ProductFilters as Filters, Product } from "@/types";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

function ProductsContent() {
  const searchParams = useSearchParams();
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<Filters>({});

  useEffect(() => {
    // Initialize filters from URL params
    const initialFilters: Filters = {};
    if (searchParams.get("category"))
      initialFilters.category = searchParams.get("category")!;
    if (searchParams.get("brand"))
      initialFilters.brand = searchParams.get("brand")!;
    if (searchParams.get("search"))
      initialFilters.search = searchParams.get("search")!;
    setFilters(initialFilters);
  }, [searchParams]);

  useEffect(() => {
    fetchProducts();
  }, [filters]);

  const fetchProducts = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== "") {
          params.append(key, value.toString());
        }
      });

      const response = await api.get(`/products/?${params.toString()}`);
      if (response.data.success) {
        setProducts(response.data.products);
      }
    } catch (error) {
      console.error("Failed to fetch products:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Products</h1>
        <p className="text-muted-foreground">
          Discover our complete collection of technology products
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-1">
          <ProductFilters filters={filters} onFiltersChange={setFilters} />
        </div>

        <div className="lg:col-span-3">
          <div className="mb-6">
            <p className="text-sm text-muted-foreground">
              {loading ? "Loading..." : `${products.length} products found`}
            </p>
          </div>
          <ProductGrid products={products} loading={loading} />
        </div>
      </div>
    </div>
  );
}

export default function ProductsPage() {
  return (
    <Suspense
      fallback={
        <div className="container py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Products</h1>
            <p className="text-muted-foreground">
              Discover our complete collection of technology products
            </p>
          </div>
          <div className="flex justify-center items-center h-64">
            <div className="text-muted-foreground">Loading products...</div>
          </div>
        </div>
      }
    >
      <ProductsContent />
    </Suspense>
  );
}
