import { useState, useEffect } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MovieThumbnail } from "./MovieThumbnail";
import MovieDataService, { type MovieData } from "../services/movieDataService";

interface SearchResultsGridProps {
  searchQuery: string;
  userId?: string;
  onMovieClick: (movieId: string) => void;
  onBackToMain: () => void;
}

export const SearchResultsGrid: React.FC<SearchResultsGridProps> = ({
  searchQuery,
  userId,
  onMovieClick,
  onBackToMain
}) => {
  const [searchResults, setSearchResults] = useState<MovieData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [totalResults, setTotalResults] = useState(0);

  useEffect(() => {
    const fetchSearchResults = async () => {
      if (!searchQuery.trim()) {
        setSearchResults([]);
        setTotalResults(0);
        return;
      }

      // Clear previous results immediately when new search starts
      setSearchResults([]);
      setIsLoading(true);
      setTotalResults(0);

      try {
        const results = await MovieDataService.searchMovies(searchQuery, 20, userId);
        setSearchResults(results.movies);
        setTotalResults(results.total);
      } catch (error) {
        console.error('Search failed:', error);
        // Fallback to empty results
        setSearchResults([]);
        setTotalResults(0);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSearchResults();
  }, [searchQuery, userId]);

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
              {isLoading ? 'Searching...' : `Found ${totalResults} movies for "${searchQuery}"`}
            </p>
          </div>
        </div>
      </div>

      {/* Grid of search results */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 max-h-[600px] overflow-y-auto pr-2">
        {isLoading ? (
          // Loading skeletons
          Array.from({length: 20}, (_, i) => (
            <MovieThumbnail
              key={`loading-${i}`}
              movie={{ id: `loading-${i}`, title: 'Loading...' }}
              isLoading={true}
              onMovieClick={onMovieClick}
            />
          ))
        ) : (
          // Actual results
          searchResults.map((movie) => (
            <MovieThumbnail
              key={movie.id}
              movie={movie}
              onMovieClick={onMovieClick}
            />
          ))
        )}
      </div>

      {/* Footer info */}
      <div className="text-center text-sm text-muted-foreground">
        {isLoading ? (
          'Loading search results...'
        ) : searchResults.length > 0 ? (
          'Click on any movie to see more details'
        ) : (
          'No results found. Try a different search term.'
        )}
      </div>
    </div>
  );
};

export default SearchResultsGrid;