import { useState, useEffect } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MovieThumbnail } from "./MovieThumbnail";
import MovieDataService, { type MovieData } from "../services/movieDataService";

interface SearchResultsGridProps {
  searchQuery: string;
  onMovieClick: (movieId: string) => void;
  onBackToMain: () => void;
}

export const SearchResultsGrid: React.FC<SearchResultsGridProps> = ({
  searchQuery,
  onMovieClick,
  onBackToMain
}) => {
  const [searchResults, setSearchResults] = useState<MovieData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasRealData, setHasRealData] = useState(false);
  const [totalResults, setTotalResults] = useState(0);

  useEffect(() => {
    const fetchSearchResults = async () => {
      if (!searchQuery.trim()) return;

      setIsLoading(true);
      try {
        const results = await MovieDataService.searchMovies(searchQuery, 20);
        setSearchResults(results.movies);
        setHasRealData(results.hasRealData);
        setTotalResults(results.total);
      } catch (error) {
        console.error('Search failed:', error);
        // Fallback to empty results
        setSearchResults([]);
        setHasRealData(false);
        setTotalResults(0);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSearchResults();
  }, [searchQuery]);

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
            <div className="flex items-center space-x-3">
              <p className="text-muted-foreground">
                {isLoading ? 'Searching...' : `Found ${totalResults} movies for "${searchQuery}"`}
              </p>
              <span className={`text-xs px-2 py-1 rounded ${hasRealData ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}`}>
                {hasRealData ? '🌐 Real Data' : '📝 Mock Data'}
              </span>
            </div>
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