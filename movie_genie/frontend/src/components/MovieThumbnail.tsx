import { useState } from "react";
import { Check, ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface MovieThumbnailProps {
  movie: {
    id: string;
    title: string;
    poster: string;
    watched?: boolean;
    liked?: boolean;
    disliked?: boolean;
  };
  showRatings?: boolean;
  onMovieClick?: (movieId: string) => void;
  onRating?: (movieId: string, rating: 'like' | 'dislike' | 'watched') => void;
}

export function MovieThumbnail({ 
  movie, 
  showRatings = false, 
  onMovieClick, 
  onRating 
}: MovieThumbnailProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Card 
      className="relative bg-gradient-to-br from-card to-card/80 border-border/50 overflow-hidden group cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-[var(--shadow-glow)]"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onMovieClick?.(movie.id)}
    >
      <div className="aspect-[2/3] bg-muted flex items-center justify-center relative">
        <img 
          src={movie.poster} 
          alt={movie.title}
          className="w-full h-full object-cover"
          onError={(e) => {
            const target = e.currentTarget as HTMLImageElement;
            target.style.display = 'none';
            const nextElement = target.nextElementSibling as HTMLElement;
            if (nextElement) {
              nextElement.style.display = 'flex';
            }
          }}
        />
        <div className="w-full h-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center text-muted-foreground text-sm hidden">
          {movie.title}
        </div>
        
        {/* Hover overlay */}
        <div className={`absolute inset-0 bg-black/60 transition-opacity duration-300 ${
          isHovered ? 'opacity-100' : 'opacity-0'
        }`}>
          <div className="absolute bottom-2 left-2 right-2">
            <h4 className="text-white text-xs font-medium line-clamp-2 mb-2">
              {movie.title}
            </h4>
            
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