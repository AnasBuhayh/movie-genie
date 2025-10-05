import { useState } from "react";
import { Link } from "react-router-dom";
import { MovieSearch } from "@/components/MovieSearch";
import { MovieDetails } from "@/components/MovieDetails";
import { RecommendationCarousel } from "@/components/RecommendationCarousel";
import { UserSelectionModal } from "@/components/UserSelectionModal";
import { SearchResultsGrid } from "@/components/SearchResultsGrid";
import { Separator } from "@/components/ui/separator";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { User, BarChart3 } from "lucide-react";

// API Hooks
import {
  usePopularMovies,
  usePersonalizedRecommendations,
  useSearchMovies,
  useMovieDetails,
  formatMovieForComponent,
} from "@/hooks/useMovieAPI";

// Import data service
import MovieDataService, { type MovieData } from "@/services/movieDataService";

// Placeholder movie for initial state
const placeholderMovie = {
  title: "Select a movie to see details",
  genres: [],
  description: "Click on any movie thumbnail to view detailed information including ratings, runtime, and more.",
  rating: undefined,
  voteCount: undefined,
  runtime: undefined,
  likes: [],
  dislikes: []
};

const Index = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMovieId, setSelectedMovieId] = useState<string>("");
  const [userId, setUserId] = useState("");
  const [isUserSelected, setIsUserSelected] = useState(false);
  const [isSearchMode, setIsSearchMode] = useState(false);

  // Movie data states
  const [popularMovies, setPopularMovies] = useState<MovieData[]>([]);
  const [personalizedMovies, setPersonalizedMovies] = useState<MovieData[]>([]);
  const [historicalMovies, setHistoricalMovies] = useState<MovieData[]>([]);
  const [watchedMovies, setWatchedMovies] = useState<MovieData[]>([]);
  const [isLoadingMovies, setIsLoadingMovies] = useState(false);

  // Movie data is now loaded in handleUserSelect, no need for separate useEffect

  // Get movie details for selected movie
  const { data: movieDetails } = useMovieDetails(selectedMovieId, !!selectedMovieId && isUserSelected);

  // Current movie for details panel
  const currentMovie = movieDetails ? {
    title: movieDetails.title || "Unknown Title",
    genres: movieDetails.genres || [],
    description: movieDetails.overview || "No description available.",
    rating: movieDetails.vote_average,
    voteCount: movieDetails.vote_count,
    runtime: movieDetails.runtime,
    likes: [],
    dislikes: []
  } : placeholderMovie;

  const handleMovieClick = (movieId: string) => {
    console.log("Movie clicked:", movieId);
    setSelectedMovieId(movieId);
  };

  const handleRating = (movieId: string, rating: 'like' | 'dislike' | 'watched') => {
    console.log("Movie rated:", movieId, rating);
    // TODO: Implement rating submission with useSubmitRating hook
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setIsSearchMode(query.trim().length > 0);
  };

  const handleBackToMain = () => {
    setSearchQuery("");
    setIsSearchMode(false);
    setSelectedMovieId("");
  };

  const handleUserSelect = async (selectedUserId: string) => {
    // Set user ID first
    setUserId(selectedUserId);
    setIsUserSelected(true);
    setIsLoadingMovies(true);

    try {
      // Load all movie data before closing modal
      const [popular, personalized, historical, watched] = await Promise.all([
        MovieDataService.getPopularMovies(10),
        MovieDataService.getPersonalizedRecommendations(selectedUserId, 10),
        MovieDataService.getHistoricalInterest(selectedUserId, 10),
        MovieDataService.getUserWatchedMovies(selectedUserId, 8),
      ]);

      setPopularMovies(popular);
      setPersonalizedMovies(personalized);
      setHistoricalMovies(historical);
      setWatchedMovies(watched);
    } catch (error) {
      console.error('Failed to load movie data:', error);
      throw error; // Propagate error to modal
    } finally {
      setIsLoadingMovies(false);
    }
  };

  const handleChangeUser = () => {
    setIsUserSelected(false);
    setUserId("");
    setSearchQuery("");
    setSelectedMovieId("");
    setIsSearchMode(false);
  };

  return (
    <>
      {/* User Selection Modal */}
      <UserSelectionModal
        isOpen={!isUserSelected}
        onUserSelect={handleUserSelect}
      />

      {/* Main App Content - only show when user is selected */}
      <div className="min-h-screen bg-background">
        <div className="container mx-auto p-6">
          {/* User Header Bar */}
          {isUserSelected && (
            <div className="flex items-center justify-between mb-6 p-4 bg-primary/5 rounded-lg">
              <div className="flex items-center space-x-3">
                <User className="h-6 w-6 text-primary" />
                <div>
                  <h2 className="text-lg font-semibold">Welcome, User {userId}!</h2>
                  <p className="text-sm text-muted-foreground">Your personalized movie recommendations</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Link to="/metrics">
                  <Button variant="outline" size="sm">
                    <BarChart3 className="mr-2 h-4 w-4" />
                    Model Metrics
                  </Button>
                </Link>
                <Button variant="outline" size="sm" onClick={handleChangeUser}>
                  Change User
                </Button>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column - Search & Movie Details */}
            <div className="lg:col-span-1 space-y-6">
              <MovieSearch onSearch={handleSearch} />
              <Separator className="border-border" />

              {/* Show movie details */}
              <MovieDetails movie={currentMovie} />
            </div>

            {/* Right Column - Search Results or Recommendations */}
            <div className="lg:col-span-2 space-y-8">
              {isSearchMode ? (
                <SearchResultsGrid
                  searchQuery={searchQuery}
                  userId={userId}
                  onMovieClick={handleMovieClick}
                  onBackToMain={handleBackToMain}
                />
              ) : (
                <>
                  {/* Popular Movies */}
                  <RecommendationCarousel
                    title="Popular Movies"
                    movies={popularMovies.map(movie => ({
                      id: movie.id,
                      title: movie.title,
                      poster: movie.poster_url || '/placeholder.svg',
                      genres: movie.genres,
                      rating: movie.rating,
                    }))}
                    onMovieClick={handleMovieClick}
                    isLoading={isLoadingMovies}
                  />

                  {/* Based on Historical Interest */}
                  <RecommendationCarousel
                    title="Based on Historical Interest"
                    movies={historicalMovies.map(movie => ({
                      id: movie.id,
                      title: movie.title,
                      poster: movie.poster_url || '/placeholder.svg',
                      genres: movie.genres,
                      rating: movie.rating,
                    }))}
                    onMovieClick={handleMovieClick}
                    isLoading={isLoadingMovies}
                  />

                  {/* You Might Like These */}
                  <RecommendationCarousel
                    title="You Might Like These"
                    movies={personalizedMovies.map(movie => ({
                      id: movie.id,
                      title: movie.title,
                      poster: movie.poster_url || '/placeholder.svg',
                      genres: movie.genres,
                      rating: movie.rating,
                    }))}
                    showRatings={true}
                    onMovieClick={handleMovieClick}
                    onRating={handleRating}
                    isLoading={isLoadingMovies}
                  />

                  {/* User's Watched Movies */}
                  <RecommendationCarousel
                    title={`Movies Watched by User ${userId}`}
                    movies={watchedMovies.map(movie => ({
                      id: movie.id,
                      title: movie.title,
                      poster: movie.poster_url || '/placeholder.svg',
                      genres: movie.genres,
                      rating: movie.rating,
                      watched: true,
                    }))}
                    onMovieClick={handleMovieClick}
                    isLoading={isLoadingMovies}
                  />
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Index;