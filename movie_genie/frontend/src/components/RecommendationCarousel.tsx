import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MovieThumbnail } from "./MovieThumbnail";
import { useRef } from "react";

interface Movie {
  id: string;
  title: string;
  poster: string;
  watched?: boolean;
  liked?: boolean;
  disliked?: boolean;
}

interface RecommendationCarouselProps {
  title: string;
  movies: Movie[];
  showRatings?: boolean;
  onMovieClick?: (movieId: string) => void;
  onRating?: (movieId: string, rating: 'like' | 'dislike' | 'watched') => void;
}

export function RecommendationCarousel({ 
  title, 
  movies, 
  showRatings = false, 
  onMovieClick, 
  onRating 
}: RecommendationCarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const scrollAmount = 200;
      scrollRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-foreground">{title}</h2>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
            onClick={() => scroll('left')}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
            onClick={() => scroll('right')}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      <div 
        ref={scrollRef}
        className="flex gap-4 overflow-x-auto scrollbar-hide pb-2"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        {movies.map((movie) => (
          <div key={movie.id} className="flex-shrink-0 w-32">
            <MovieThumbnail
              movie={movie}
              showRatings={showRatings}
              onMovieClick={onMovieClick}
              onRating={onRating}
            />
          </div>
        ))}
      </div>
    </div>
  );
}