import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface Movie {
  id: string;
  title: string;
  poster: string;
  genres?: string[];
  description?: string;
  rating?: number;
}

interface SearchResultsGridProps {
  searchQuery: string;
  onMovieClick: (movieId: string) => void;
  onBackToMain: () => void;
}

// Mock search results - 20 placeholder movies
const createMockSearchResult = (id: number, query: string) => ({
  id: id.toString(),
  title: `${query} Result ${id}`,
  poster: '/placeholder.svg',
  genres: ['Action', 'Drama', 'Thriller'],
  description: `A great movie related to "${query}" with an engaging storyline.`,
  rating: 7.0 + Math.random() * 2.5,
});

export const SearchResultsGrid: React.FC<SearchResultsGridProps> = ({
  searchQuery,
  onMovieClick,
  onBackToMain
}) => {
  // Generate 20 mock search results based on the query
  const searchResults = Array.from({length: 20}, (_, i) =>
    createMockSearchResult(i + 1, searchQuery)
  );

  return (
    <div className="space-y-6">
      {/* Header with back button */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button
            variant="outline"
            size="sm"
            onClick={onBackToMain}
            className="flex items-center space-x-2"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Back to Main</span>
          </Button>
          <div>
            <h2 className="text-2xl font-bold">Search Results</h2>
            <p className="text-muted-foreground">
              Found {searchResults.length} movies for "{searchQuery}"
            </p>
          </div>
        </div>
      </div>

      {/* Grid of search results */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 max-h-[600px] overflow-y-auto pr-2">
        {searchResults.map((movie) => (
          <Card
            key={movie.id}
            className="relative bg-gradient-to-br from-card to-card/80 border-border/50 overflow-hidden group cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-lg"
            onClick={() => onMovieClick(movie.id)}
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
              <div className="absolute inset-0 bg-black/60 transition-opacity duration-300 opacity-0 group-hover:opacity-100">
                <div className="absolute bottom-0 left-0 right-0 p-3 text-white">
                  <h3 className="font-semibold text-sm line-clamp-2 mb-1">{movie.title}</h3>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-yellow-400">★ {movie.rating?.toFixed(1)}</span>
                    <span className="text-gray-300">{movie.genres?.[0]}</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Footer info */}
      <div className="text-center text-sm text-muted-foreground">
        Click on any movie to see more details
      </div>
    </div>
  );
};

export default SearchResultsGrid;