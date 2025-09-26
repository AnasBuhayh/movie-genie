/**
 * API configuration and service functions for Movie Genie Flask backend
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export const API_ENDPOINTS = {
  // Search endpoints
  SEARCH: `${API_BASE_URL}/search`,
  SEMANTIC_SEARCH: `${API_BASE_URL}/search/semantic`,

  // Movie endpoints
  MOVIE_DETAILS: (id: string) => `${API_BASE_URL}/movies/${id}`,
  POPULAR_MOVIES: `${API_BASE_URL}/movies/popular`,

  // Recommendation endpoints
  RECOMMENDATIONS: `${API_BASE_URL}/recommendations`,
  PERSONALIZED_RECOMMENDATIONS: `${API_BASE_URL}/recommendations/personalized`,

  // User feedback endpoints
  FEEDBACK: `${API_BASE_URL}/feedback`,
  RATING: `${API_BASE_URL}/rating`,

  // User endpoints
  USER_INFO: `${API_BASE_URL}/users/info`,
  USER_PROFILE: (userId: number) => `${API_BASE_URL}/users/${userId}/profile`,
} as const;

// API Response Types
export interface Movie {
  movieId: number;
  title: string;
  overview: string;
  genres: string[];
  release_date?: string;
  vote_average?: number;
  vote_count?: number;
  poster_path?: string;
  backdrop_path?: string;
  runtime?: number;
  similarity_score?: number;
  personalized_score?: number;
  rank?: number;
}

export interface SearchResponse {
  movies: Movie[];
  total: number;
  query: string;
  search_type: 'semantic' | 'traditional';
}

export interface RecommendationResponse {
  movies: Movie[];
  recommendation_type: 'popular' | 'personalized' | 'content_based';
  user_context?: any;
}

export interface FeedbackRequest {
  user_id?: string;
  movie_id: number;
  rating: number;
  feedback_type: 'like' | 'dislike' | 'rating';
}

export interface UserInfo {
  user_id_range: {
    min: number;
    max: number;
    total: number;
  };
  interaction_stats: {
    mean_interactions_per_user: number;
    min_interactions: number;
    max_interactions: number;
    median_interactions: number;
  };
  sample_user_ids: number[];
  instructions: {
    valid_range: string;
    note: string;
  };
}

export interface UserProfile {
  user_id: number;
  statistics: {
    total_interactions: number;
    unique_movies_watched: number;
  };
  interaction_history: Array<{
    movieId: number;
    title?: string;
    rating?: number;
    timestamp?: string;
  }>;
  total_history_length: number;
  recommendation_ready: boolean;
}

// API Service Functions
export class MovieGenieAPI {
  private static async fetchAPI<T>(url: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  // Search Movies
  static async searchMovies(query: string, useSemanticSearch = true): Promise<SearchResponse> {
    const endpoint = useSemanticSearch ? API_ENDPOINTS.SEMANTIC_SEARCH : API_ENDPOINTS.SEARCH;
    const url = `${endpoint}?q=${encodeURIComponent(query)}`;

    return this.fetchAPI<SearchResponse>(url);
  }

  // Get Movie Details
  static async getMovieDetails(movieId: string): Promise<Movie> {
    return this.fetchAPI<Movie>(API_ENDPOINTS.MOVIE_DETAILS(movieId));
  }

  // Get Popular Movies
  static async getPopularMovies(limit = 20): Promise<RecommendationResponse> {
    const url = `${API_ENDPOINTS.POPULAR_MOVIES}?limit=${limit}`;
    return this.fetchAPI<RecommendationResponse>(url);
  }

  // Get Personalized Recommendations
  static async getPersonalizedRecommendations(
    userId?: string,
    interactionHistory?: any[]
  ): Promise<RecommendationResponse> {
    const payload = {
      user_id: userId,
      interaction_history: interactionHistory,
    };

    return this.fetchAPI<RecommendationResponse>(API_ENDPOINTS.PERSONALIZED_RECOMMENDATIONS, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Submit User Feedback
  static async submitFeedback(feedback: FeedbackRequest): Promise<{ success: boolean; message: string }> {
    return this.fetchAPI(API_ENDPOINTS.FEEDBACK, {
      method: 'POST',
      body: JSON.stringify(feedback),
    });
  }

  // Submit Movie Rating
  static async submitRating(
    movieId: number,
    rating: number,
    userId?: string
  ): Promise<{ success: boolean; message: string }> {
    const payload = {
      user_id: userId,
      movie_id: movieId,
      rating,
      feedback_type: 'rating' as const,
    };

    return this.fetchAPI(API_ENDPOINTS.RATING, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Get Content-Based Recommendations for a movie
  static async getContentBasedRecommendations(
    movieId: number,
    limit = 10
  ): Promise<RecommendationResponse> {
    const url = `${API_ENDPOINTS.RECOMMENDATIONS}?movie_id=${movieId}&limit=${limit}&type=content_based`;
    return this.fetchAPI<RecommendationResponse>(url);
  }

  // Get User Information
  static async getUserInfo(): Promise<UserInfo> {
    const response = await this.fetchAPI<{data: UserInfo, success: boolean, message: string}>(API_ENDPOINTS.USER_INFO);
    return response.data;
  }

  // Get User Profile
  static async getUserProfile(userId: number): Promise<UserProfile> {
    return this.fetchAPI<UserProfile>(API_ENDPOINTS.USER_PROFILE(userId));
  }
}

export default MovieGenieAPI;