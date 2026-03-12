import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { useAuth } from "@/context/AuthContext";
import api from "@/lib/api";
import { formatPrice } from "@/lib/utils";
import { Product } from "@/types";
import { Heart, Share2, ShoppingCart, Star } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import { toast } from "sonner";

interface ProductCardProps {
  product: Product;
  size?: "default" | "compact";
}

export function ProductCard({ product, size = "default" }: ProductCardProps) {
  const { user } = useAuth();
  const [imageError, setImageError] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [likesCount, setLikesCount] = useState(0);

  // Add safety checks for product data
  if (!product || !product.id) {
    return (
      <Card className="h-full flex items-center justify-center p-4">
        <p className="text-sm text-muted-foreground">Product unavailable</p>
      </Card>
    );
  }

  const isCompact = size === "compact";

  useEffect(() => {
    if (user && product.id) {
      checkLikeStatus();
    }
    fetchLikesCount();
  }, [user, product.id]);

  const checkLikeStatus = async () => {
    if (!user) return;

    try {
      const response = await api.post("/likes/check", {
        product_id: product.id,
      });
      if (response.data.success) {
        setIsLiked(response.data.is_liked);
      }
    } catch (error) {
      // Silently handle error - likes are not critical
    }
  };

  const fetchLikesCount = async () => {
    try {
      const response = await api.get(`/likes/product/${product.id}`);
      if (response.data.success) {
        setLikesCount(response.data.likes_count);
      }
    } catch (error) {
      // Silently handle error
    }
  };

  const handleLike = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!user) {
      toast.error("Please login to like products");
      return;
    }

    try {
      const response = await api.post("/likes/toggle", {
        product_id: product.id,
      });

      if (response.data.success) {
        setIsLiked(response.data.is_liked);
        setLikesCount((prev) => (response.data.is_liked ? prev + 1 : prev - 1));
        toast(response.data.message);
      }
    } catch (error) {
      toast.error("Failed to update like status");
    }
  };

  const handleShare = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const shareData = {
      title: product.name,
      text: `Check out this amazing ${product.name} from ${product.brand}!`,
      url: window.location.origin + `/products/${product.id}`,
    };

    try {
      if (navigator.share && navigator.canShare(shareData)) {
        await navigator.share(shareData);
      } else {
        // Fallback to clipboard
        await navigator.clipboard.writeText(shareData.url);
        toast("Product link copied to clipboard!");
      }
    } catch (error) {
      // Fallback to clipboard
      try {
        await navigator.clipboard.writeText(shareData.url);
        toast("Product link copied to clipboard!");
      } catch (clipboardError) {
        toast.error("Failed to share product");
      }
    }
  };

  const handleAddToCart = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!user) {
      toast.error("Please login to add items to cart");
      return;
    }

    try {
      await api.post("/cart/add", {
        user_id: user.id,
        product_id: product.id,
        quantity: 1,
      });
      toast("Added to cart!");
    } catch (error) {
      toast.error("Failed to add to cart");
    }
  };

  return (
    <Link href={`/products/${product.id}`}>
      <Card className="group overflow-hidden hover:shadow-lg transition-all duration-300 h-full">
        <div
          className={`relative overflow-hidden ${
            isCompact ? "aspect-[4/3]" : "aspect-square"
          }`}
        >
          <Image
            src={
              imageError
                ? "/placeholder-image.svg"
                : product.imageUrl || "/placeholder-image.svg"
            }
            alt={product.name || "Product"}
            fill
            className="object-cover group-hover:scale-105 transition-transform duration-300"
            onError={() => setImageError(true)}
          />
          {product.isOnSale && product.salePercentage && (
            <Badge
              className={`absolute top-2 left-2 bg-red-500 hover:bg-red-600 ${
                isCompact ? "text-xs" : ""
              }`}
            >
              -{product.salePercentage}%
            </Badge>
          )}
          {product.inStock === false && (
            <Badge
              variant="secondary"
              className={`absolute top-2 right-2 ${isCompact ? "text-xs" : ""}`}
            >
              Out of Stock
            </Badge>
          )}
        </div>

        <CardContent className={`space-y-2 ${isCompact ? "p-3" : "p-4"}`}>
          <div className="space-y-1">
            <h3
              className={`font-semibold line-clamp-2 group-hover:text-primary transition-colors ${
                isCompact ? "text-sm" : ""
              }`}
            >
              {product.name || "Unnamed Product"}
            </h3>
            <p
              className={`text-muted-foreground ${
                isCompact ? "text-xs" : "text-sm"
              }`}
            >
              {product.brand || "Unknown Brand"}
            </p>
          </div>

          <div className="flex items-center space-x-1">
            <Star
              className={`fill-yellow-400 text-yellow-400 ${
                isCompact ? "h-3 w-3" : "h-4 w-4"
              }`}
            />
            <span
              className={`font-medium ${isCompact ? "text-xs" : "text-sm"}`}
            >
              {product.rating || 0}
            </span>
            <span
              className={`text-muted-foreground ${
                isCompact ? "text-xs" : "text-sm"
              }`}
            >
              ({product.reviewCount || 0})
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <span
              className={`font-bold ${isCompact ? "text-base" : "text-lg"}`}
            >
              {formatPrice(product.price || 0)}
            </span>
            {product.originalPrice &&
              product.originalPrice > (product.price || 0) && (
                <span
                  className={`text-muted-foreground line-through ${
                    isCompact ? "text-xs" : "text-sm"
                  }`}
                >
                  {formatPrice(product.originalPrice)}
                </span>
              )}
          </div>
        </CardContent>

        <CardFooter className={`pt-0 ${isCompact ? "p-3" : "p-4"}`}>
          <div className="flex space-x-2 w-full">
            <Button
              className={`flex-1 ${isCompact ? "h-8 text-xs" : ""}`}
              onClick={handleAddToCart}
              disabled={product.inStock === false}
            >
              {!isCompact && (
                <ShoppingCart
                  className={`mr-2 ${isCompact ? "h-3 w-3" : "h-4 w-4"}`}
                />
              )}

              {product.inStock !== false ? "Add to Cart" : "Out of Stock"}
            </Button>

            <Button
              size={isCompact ? "sm" : "default"}
              variant="outline"
              onClick={handleLike}
              className={`px-3 ${isLiked ? "text-red-500 border-red-200" : ""}`}
            >
              <Heart
                className={`${isCompact ? "h-3 w-3" : "h-4 w-4"} ${
                  isLiked ? "fill-red-500" : ""
                }`}
              />
              {!isCompact && likesCount > 0 && (
                <span className="ml-1 text-xs">{likesCount}</span>
              )}
            </Button>

            <Button
              size={isCompact ? "sm" : "default"}
              variant="outline"
              onClick={handleShare}
              className="px-3"
            >
              <Share2 className={`${isCompact ? "h-3 w-3" : "h-4 w-4"}`} />
            </Button>
          </div>
        </CardFooter>
      </Card>
    </Link>
  );
}
