"use client";

import { ProductGrid } from "@/components/products/ProductGrid";
import { Button } from "@/components/ui/button";
import { PixelAnimation } from "@/components/ui/PixelAnimation";
import api from "@/lib/api";
import { Product } from "@/types";
import { ArrowRight, Bot, ShoppingBag } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function HomePage() {
  const [featuredProducts, setFeaturedProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFeaturedProducts();
  }, []);

  const fetchFeaturedProducts = async () => {
    try {
      const response = await api.get("/products/", {
        params: { limit: 8, min_rating: 4.5 },
      });
      if (response.data.success) {
        setFeaturedProducts(response.data.products);
      }
    } catch (error) {
      console.error("Failed to fetch featured products:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-16">
      {" "}
      {/* Hero Section */}
      <section className="relative py-20 px-4 text-center overflow-hidden">
        <PixelAnimation className="z-0" opacity={1} speed={1} />
        <div className="absolute inset-0 bg-gradient-radial from-black via-black/60 to-transparent z-5"></div>
        <div className="container max-w-4xl relative z-10">
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Shop Smarter with AI
          </h1>
          <p className="text-xl text-foreground mb-8 max-w-2xl mx-auto">
            Discover the perfect tech products with our intelligent shopping
            assistant. Get personalized recommendations and expert advice.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/products">
                <ShoppingBag className="mr-2 h-5 w-5" />
                Browse Products
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="/chat">
                <Bot className="mr-2 h-5 w-5" />
                Try AI Assistant
              </Link>
            </Button>
          </div>
        </div>
      </section>
      {/* Features Section */}
      {/* <section className="container px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Why Choose S-TORE?</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Experience the future of online shopping with our cutting-edge
            features
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <Bot className="h-6 w-6 text-primary" />
              </div>
              <CardTitle>AI Shopping Assistant</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Get personalized product recommendations and instant answers to
                your questions
              </p>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <Star className="h-6 w-6 text-primary" />
              </div>
              <CardTitle>Premium Quality</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Curated selection of top-rated electronics from trusted brands
              </p>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <Zap className="h-6 w-6 text-primary" />
              </div>
              <CardTitle>Fast & Secure</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Lightning-fast checkout and secure payment processing
              </p>
            </CardContent>
          </Card>
        </div>
      </section> */}
      {/* Featured Products */}
      <section className="container px-4">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold mb-2">Featured Products</h2>
            <p className="text-muted-foreground">
              Discover our top-rated electronics and latest arrivals
            </p>
          </div>
          <Button variant="outline" asChild>
            <Link href="/products">
              View All
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>

        <ProductGrid products={featuredProducts} loading={loading} />
      </section>{" "}
      {/* CTA Section */}
      <section className="bg-background py-16 relative overflow-hidden">
        <PixelAnimation className="z-0" opacity={1} speed={1} />
        {/* Radial gradient overlay for better text visibility */}
        <div className="absolute inset-0 bg-gradient-radial from-black via-black/50 to-transparent z-5"></div>
        <div className="container px-4 text-center relative z-10">
          <h2 className="text-3xl font-bold mb-4">Ready to Start Shopping?</h2>
          <p className="text-foreground mb-8 max-w-2xl mx-auto">
            Join the satisfied customers who trust S-TORE for their technology
            needs
          </p>
          <Button size="lg" asChild>
            <Link href="/chat">
              <Bot className="h-5 w-5" />
              Chat with AI Assistant
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
}
