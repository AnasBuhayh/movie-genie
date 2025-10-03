import { useState, useEffect } from "react";
import { MovieSearch } from "@/components/MovieSearch";
import { MovieDetails } from "@/components/MovieDetails";
import { RecommendationCarousel } from "@/components/RecommendationCarousel";
import { UserSelectionModal } from "@/components/UserSelectionModal";
import { SearchResultsGrid } from "@/components/SearchResultsGrid";
import { Separator } from "@/components/ui/separator";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { User } from "lucide-react";

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
  title: "Search for a movie to see details",
  genres: [""],
  description: "Use the search above to find movies and get personalized recommendations based on your preferences.",
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

  // Load movie data when user is selected
  useEffect(() => {
    if (!isUserSelected) return;

    const loadMovieData = async () => {
      setIsLoadingMovies(true);
      try {
        const [popular, personalized, historical, watched] = await Promise.all([
          MovieDataService.getPopularMovies(10),
          MovieDataService.getPersonalizedRecommendations(userId, 10),
          MovieDataService.getHistoricalInterest(userId, 10),
          MovieDataService.getUserWatchedMovies(userId, 8),
        ]);

        setPopularMovies(popular);
        setPersonalizedMovies(personalized);
        setHistoricalMovies(historical);
        setWatchedMovies(watched);
      } catch (error) {
        console.error('Failed to load movie data:', error);
      } finally {
        setIsLoadingMovies(false);
      }
    };

    loadMovieData();
  }, [isUserSelected, userId]);

  // Get movie details for selected movie
  const { data: movieDetails } = useMovieDetails(selectedMovieId, !!selectedMovieId && isUserSelected);

  // Current movie for details panel
  const currentMovie = movieDetails ? {
    title: movieDetails.title,
    genres: movieDetails.genres || [],
    description: movieDetails.overview,
    likes: [
      `Rating: ${movieDetails.vote_average}/10`,
      `${movieDetails.vote_count} votes`,
      ...(movieDetails.runtime ? [`Runtime: ${movieDetails.runtime} minutes`] : [])
    ],
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

  const handleUserSelect = (selectedUserId: string) => {
    setUserId(selectedUserId);
    setIsUserSelected(true);
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
              <Button variant="outline" size="sm" onClick={handleChangeUser}>
                Change User
              </Button>
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