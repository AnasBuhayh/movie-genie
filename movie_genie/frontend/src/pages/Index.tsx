import { useState } from "react";
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

// Mock movie data with placeholders
const createMockMovie = (id: number, title: string) => ({
  id: id.toString(),
  title,
  poster: '/placeholder.svg',
  genres: ['Action', 'Drama'],
  description: `${title} - A great movie with an engaging storyline.`,
  rating: 7.5 + Math.random() * 2,
});

const mockMovies = {
  popular: Array.from({length: 10}, (_, i) => createMockMovie(i + 1, `Popular Movie ${i + 1}`)),
  historical: Array.from({length: 10}, (_, i) => createMockMovie(i + 11, `Historical Interest ${i + 1}`)),
  recommended: Array.from({length: 10}, (_, i) => createMockMovie(i + 21, `Recommended ${i + 1}`)),
  watched: Array.from({length: 8}, (_, i) => createMockMovie(i + 31, `Watched Movie ${i + 1}`)),
};

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

  // API Queries - only run when user is selected
  const { data: popularData, isLoading: isLoadingPopular } = usePopularMovies();
  const { data: personalizedData, isLoading: isLoadingPersonalized } = usePersonalizedRecommendations(
    isUserSelected && userId ? userId : undefined,
    undefined,
    isUserSelected && userId.length > 0
  );
  const { data: searchData, isLoading: isSearching } = useSearchMovies(searchQuery, searchQuery.length > 0 && isUserSelected);
  const { data: movieDetails } = useMovieDetails(selectedMovieId, !!selectedMovieId && isUserSelected);

  // Transform API data for components
  const popularMovies = popularData?.movies?.map(formatMovieForComponent) || [];
  const personalizedMovies = personalizedData?.movies?.map(formatMovieForComponent) || [];
  const searchResults = searchData?.movies?.map(formatMovieForComponent) || [];

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

              {/* Show search results or movie details */}
              {isSearching ? (
                <Card>
                  <CardContent className="p-4">
                    <Skeleton className="h-4 w-full mb-2" />
                    <Skeleton className="h-4 w-3/4 mb-2" />
                    <Skeleton className="h-20 w-full" />
                  </CardContent>
                </Card>
              ) : searchResults.length > 0 ? (
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold">Search Results</h3>
                  {searchResults.slice(0, 5).map((movie) => (
                    <Card key={movie.id} className="cursor-pointer hover:bg-accent" onClick={() => handleMovieClick(movie.id)}>
                      <CardContent className="p-3">
                        <h4 className="font-medium">{movie.title}</h4>
                        <p className="text-sm text-muted-foreground">{movie.genres?.join(", ")}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <MovieDetails movie={currentMovie} />
              )}
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
                    movies={mockMovies.popular}
                    onMovieClick={handleMovieClick}
                  />

                  {/* Based on Historical Interest */}
                  <RecommendationCarousel
                    title="Based on Historical Interest"
                    movies={mockMovies.historical}
                    onMovieClick={handleMovieClick}
                  />

                  {/* You Might Like These */}
                  <RecommendationCarousel
                    title="You Might Like These"
                    movies={mockMovies.recommended}
                    showRatings={true}
                    onMovieClick={handleMovieClick}
                    onRating={handleRating}
                  />

                  {/* User's Watched Movies */}
                  <RecommendationCarousel
                    title={`Movies Watched by User ${userId}`}
                    movies={mockMovies.watched}
                    onMovieClick={handleMovieClick}
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