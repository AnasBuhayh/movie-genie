/**
 * React Query hooks for Movie Genie API integration
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { MovieGenieAPI, Movie, SearchResponse, RecommendationResponse, FeedbackRequest } from '@/lib/api';

// Query Keys
export const QUERY_KEYS = {
  MOVIES: ['movies'] as const,
  SEARCH: (query: string) => ['movies', 'search', query] as const,
  MOVIE_DETAILS: (id: string) => ['movies', 'details', id] as const,
  POPULAR: ['movies', 'popular'] as const,
  RECOMMENDATIONS: (userId?: string) => ['recommendations', userId] as const,
  CONTENT_BASED: (movieId: number) => ['recommendations', 'content', movieId] as const,
} as const;

// Search Movies Hook
export function useSearchMovies(query: string, enabled = true) {
  return useQuery({
    queryKey: QUERY_KEYS.SEARCH(query),
    queryFn: () => MovieGenieAPI.searchMovies(query, true), // Use semantic search by default
    enabled: enabled && query.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Movie Details Hook
export function useMovieDetails(movieId: string, enabled = true) {
  return useQuery({
    queryKey: QUERY_KEYS.MOVIE_DETAILS(movieId),
    queryFn: () => MovieGenieAPI.getMovieDetails(movieId),
    enabled: enabled && !!movieId,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

// Popular Movies Hook
export function usePopularMovies(limit = 20) {
  return useQuery({
    queryKey: QUERY_KEYS.POPULAR,
    queryFn: () => MovieGenieAPI.getPopularMovies(limit),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

// Personalized Recommendations Hook
export function usePersonalizedRecommendations(
  userId?: string,
  interactionHistory?: any[],
  enabled = true
) {
  return useQuery({
    queryKey: QUERY_KEYS.RECOMMENDATIONS(userId),
    queryFn: () => MovieGenieAPI.getPersonalizedRecommendations(userId, interactionHistory),
    enabled: enabled,
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
}

// Content-Based Recommendations Hook
export function useContentBasedRecommendations(movieId: number, limit = 10, enabled = true) {
  return useQuery({
    queryKey: QUERY_KEYS.CONTENT_BASED(movieId),
    queryFn: () => MovieGenieAPI.getContentBasedRecommendations(movieId, limit),
    enabled: enabled && !!movieId,
    staleTime: 20 * 60 * 1000, // 20 minutes
  });
}

// Submit Feedback Mutation
export function useSubmitFeedback() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (feedback: FeedbackRequest) => MovieGenieAPI.submitFeedback(feedback),
    onSuccess: (data, variables) => {
      // Invalidate recommendations when user provides feedback
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });

      console.log('Feedback submitted successfully:', data);
    },
    onError: (error) => {
      console.error('Failed to submit feedback:', error);
    },
  });
}

// Submit Rating Mutation
export function useSubmitRating() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ movieId, rating, userId }: { movieId: number; rating: number; userId?: string }) =>
      MovieGenieAPI.submitRating(movieId, rating, userId),
    onSuccess: (data, variables) => {
      // Invalidate recommendations when user rates a movie
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });

      console.log('Rating submitted successfully:', data);
    },
    onError: (error) => {
      console.error('Failed to submit rating:', error);
    },
  });
}

// Utility function to convert API movie data to component format
export function formatMovieForComponent(apiMovie: Movie) {
  return {
    id: apiMovie.movieId.toString(),
    title: apiMovie.title,
    poster: apiMovie.poster_path || '/placeholder-poster.jpg',
    genres: apiMovie.genres,
    description: apiMovie.overview,
    rating: apiMovie.vote_average,
    similarity_score: apiMovie.similarity_score,
    personalized_score: apiMovie.personalized_score,
    rank: apiMovie.rank,
  };
}

// Search suggestions hook (for autocomplete)
export function useSearchSuggestions(query: string, enabled = true) {
  return useQuery({
    queryKey: ['search', 'suggestions', query],
    queryFn: async () => {
      if (!query || query.length < 2) return [];

      // For now, return empty array - implement when Flask backend has suggestions endpoint
      return [];
    },
    enabled: enabled && query.length >= 2,
    staleTime: 5 * 60 * 1000,
  });
}