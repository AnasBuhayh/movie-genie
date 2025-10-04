import { useState } from "react";
import { Check, ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface MovieThumbnailProps {
  movie: {
    id: string;
    title: string;
    poster?: string | null;  // Optional - can be null/undefined for real data
    poster_url?: string | null;  // Alternative API field name
    genres?: string[];
    rating?: number;
    vote_average?: number;
    overview?: string;
    watched?: boolean;
    liked?: boolean;
    disliked?: boolean;
  };
  showRatings?: boolean;
  isLoading?: boolean;
  onMovieClick?: (movieId: string) => void;
  onRating?: (movieId: string, rating: 'like' | 'dislike' | 'watched') => void;
}

export function MovieThumbnail({
  movie,
  showRatings = false,
  isLoading = false,
  onMovieClick,
  onRating
}: MovieThumbnailProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [imageError, setImageError] = useState(false);

  // Smart poster URL resolution - try multiple sources
  const getPosterUrl = (): string | null => {
    return movie.poster_url || movie.poster || null;
  };

  const posterUrl = getPosterUrl();
  const shouldShowImage = posterUrl && !imageError && !isLoading;
  const displayRating = movie.vote_average || movie.rating;

  // Loading state
  if (isLoading) {
    return (
      <Card className="relative bg-gradient-to-br from-card to-card/80 border-border/50 overflow-hidden">
        <div className="aspect-[2/3] bg-muted animate-pulse flex items-center justify-center">
          <div className="text-muted-foreground text-sm">Loading...</div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      className="relative bg-gradient-to-br from-card to-card/80 border-border/50 overflow-hidden group cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-[var(--shadow-glow)]"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onMovieClick?.(movie.id)}
    >
      <div className="aspect-[2/3] bg-muted flex items-center justify-center relative">
        {shouldShowImage ? (
          <img
            src={posterUrl}
            alt={movie.title}
            className="w-full h-full object-cover"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center p-4">
            <p className="text-muted-foreground text-sm text-center line-clamp-3">
              {movie.title || 'No Title'}
            </p>
          </div>
        )}

        {/* Always show title overlay at bottom when image is present */}
        {shouldShowImage && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
            <h4 className="text-white text-xs font-medium line-clamp-2">
              {movie.title}
            </h4>
          </div>
        )}
        
        {/* Hover overlay */}
        <div className={`absolute inset-0 bg-black/60 transition-opacity duration-300 ${
          isHovered ? 'opacity-100' : 'opacity-0'
        }`}>
          <div className="absolute bottom-2 left-2 right-2">
            <h4 className="text-white text-xs font-medium line-clamp-2 mb-1">
              {movie.title}
            </h4>

            {/* Show rating if available */}
            {displayRating && (
              <div className="text-yellow-400 text-xs mb-1">
                ★ {displayRating.toFixed(1)}
              </div>
            )}

            {/* Show genres if available */}
            {movie.genres && movie.genres.length > 0 && (
              <div className="text-gray-300 text-xs mb-2">
                {movie.genres.slice(0, 2).join(", ")}
              </div>
            )}
            
            {showRatings && (
              <div className="flex justify-between items-center gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  className={`h-6 w-6 p-0 ${movie.watched ? 'text-cinema-gold' : 'text-white/70'} hover:text-cinema-gold`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRating?.(movie.id, 'watched');
                  }}
                >
                  <Check className="h-3 w-3" />
                </Button>
                
                <Button
                  size="sm"
                  variant="ghost"
                  className={`h-6 w-6 p-0 ${movie.liked ? 'text-rating-positive' : 'text-white/70'} hover:text-rating-positive`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRating?.(movie.id, 'like');
                  }}
                >
                  <ThumbsUp className="h-3 w-3" />
                </Button>
                
                <Button
                  size="sm"
                  variant="ghost"
                  className={`h-6 w-6 p-0 ${movie.disliked ? 'text-rating-negative' : 'text-white/70'} hover:text-rating-negative`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRating?.(movie.id, 'dislike');
                  }}
                >
                  <ThumbsDown className="h-3 w-3" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}