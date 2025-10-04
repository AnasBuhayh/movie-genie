/**
 * Movie Data Service - Handles both real API data and mock/placeholder data
 * Provides seamless switching between data sources for gradual API integration
 */

import { MovieGenieAPI } from '../lib/api';

// Standardized movie data interface
export interface MovieData {
  id: string;
  title: string;
  poster_url?: string | null;
  genres?: string[];
  rating?: number;
  vote_average?: number;
  overview?: string;
  release_date?: string;
  runtime?: number;
  watched?: boolean;
  liked?: boolean;
  disliked?: boolean;
}

// Search response interface
export interface SearchResults {
  movies: MovieData[];
  total: number;
  query: string;
  hasRealData: boolean;
}

// Configuration for data sources
// Use real API by default, allow override via env vars
const DATA_SOURCE_CONFIG = {
  popular: import.meta.env.VITE_USE_REAL_POPULAR !== 'false',
  search: import.meta.env.VITE_USE_REAL_SEARCH !== 'false',
  recommendations: import.meta.env.VITE_USE_REAL_RECOMMENDATIONS !== 'false',
  movieDetails: import.meta.env.VITE_USE_REAL_MOVIE_DETAILS !== 'false',
};

export class MovieDataService {
  // Generate mock movie data
  private static createMockMovie(id: number, title: string, category?: string): MovieData {
    const categories = ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance', 'Adventure'];
    const randomGenres = categories.sort(() => 0.5 - Math.random()).slice(0, 2);

    return {
      id: id.toString(),
      title: title || `${category || 'Movie'} ${id}`,
      poster_url: null, // Will trigger fallback to gradient
      genres: randomGenres,
      rating: 7.0 + Math.random() * 2.5,
      overview: `${title} - A great movie with an engaging storyline and excellent performances.`,
      release_date: `202${Math.floor(Math.random() * 4)}-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-15`,
      runtime: 90 + Math.floor(Math.random() * 60),
    };
  }

  // Transform API data to standardized format
  private static transformApiMovie(apiMovie: any): MovieData {
    return {
      id: apiMovie.movieId?.toString() || apiMovie.id?.toString(),
      title: apiMovie.title,
      poster_url: apiMovie.poster_path ? `https://image.tmdb.org/t/p/w500${apiMovie.poster_path}` : null,
      genres: apiMovie.genres || [],
      rating: apiMovie.vote_average,
      vote_average: apiMovie.vote_average,
      overview: apiMovie.overview,
      release_date: apiMovie.release_date,
      runtime: apiMovie.runtime,
    };
  }

  // Get popular movies (with fallback)
  static async getPopularMovies(limit: number = 20): Promise<MovieData[]> {
    if (DATA_SOURCE_CONFIG.popular) {
      try {
        console.log('🔄 Attempting to fetch real popular movies...');
        const response = await MovieGenieAPI.getPopularMovies(limit);
        const movies = response.movies?.map(this.transformApiMovie) || [];
        console.log('✅ Got real popular movies:', movies.length);
        return movies;
      } catch (error) {
        console.warn('⚠️ Real popular movies failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock popular movies');
    return Array.from({length: limit}, (_, i) =>
      this.createMockMovie(i + 1, `Popular Movie ${i + 1}`, 'Popular')
    );
  }

  // Get personalized recommendations (with fallback)
  static async getPersonalizedRecommendations(userId?: string, limit: number = 20): Promise<MovieData[]> {
    if (DATA_SOURCE_CONFIG.recommendations && userId) {
      try {
        console.log('🔄 Attempting to fetch real recommendations for user:', userId);

        // Fetch user's interaction history to personalize recommendations
        let interactionHistory: any[] = [];
        try {
          const userProfile = await MovieGenieAPI.getUserProfile(parseInt(userId));
          interactionHistory = userProfile.interaction_history || [];
          console.log(`📊 Got ${interactionHistory.length} user interactions for personalization`);
        } catch (error) {
          console.warn('⚠️ Could not fetch user history, using basic recommendations:', error);
        }

        const response = await MovieGenieAPI.getPersonalizedRecommendations(userId, interactionHistory);
        const movies = response.movies?.map(this.transformApiMovie) || [];
        console.log('✅ Got real recommendations:', movies.length);
        return movies;
      } catch (error) {
        console.warn('⚠️ Real recommendations failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock recommendations');
    return Array.from({length: limit}, (_, i) =>
      this.createMockMovie(i + 100, `Recommended Movie ${i + 1}`, 'Recommended')
    );
  }

  // Search movies (no fallback - real data only)
  static async searchMovies(query: string, limit: number = 20, userId?: string): Promise<SearchResults> {
    if (!query.trim()) {
      return {
        movies: [],
        total: 0,
        query,
        hasRealData: false,
      };
    }

    console.log('🔄 Searching for:', query, userId ? `(user: ${userId})` : '');

    // Try semantic search first
    try {
      const response = await MovieGenieAPI.searchMovies(query, true, userId);
      const movies = response.movies?.map(this.transformApiMovie) || [];
      console.log('✅ Semantic search results:', movies.length);

      return {
        movies: movies.slice(0, limit),
        total: movies.length,
        query,
        hasRealData: true,
      };
    } catch (semanticError) {
      console.warn('⚠️ Semantic search failed, trying traditional search:', semanticError);

      // Fallback to traditional search
      try {
        const response = await MovieGenieAPI.searchMovies(query, false, userId);
        const movies = response.movies?.map(this.transformApiMovie) || [];
        console.log('✅ Traditional search results:', movies.length);

        return {
          movies: movies.slice(0, limit),
          total: movies.length,
          query,
          hasRealData: true,
        };
      } catch (traditionalError) {
        console.error('❌ All search methods failed:', traditionalError);
        throw new Error(`Search failed: ${traditionalError instanceof Error ? traditionalError.message : String(traditionalError)}`);
      }
    }
  }

  // Get movie details (with fallback)
  static async getMovieDetails(movieId: string): Promise<MovieData | null> {
    if (DATA_SOURCE_CONFIG.movieDetails) {
      try {
        console.log('🔄 Attempting to fetch real movie details for:', movieId);
        const movie = await MovieGenieAPI.getMovieDetails(movieId);
        console.log('✅ Got real movie details');
        return this.transformApiMovie(movie);
      } catch (error) {
        console.warn('⚠️ Real movie details failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock movie details for:', movieId);
    return this.createMockMovie(parseInt(movieId) || 1, `Movie Details ${movieId}`, 'Details');
  }

  // Get historical interest movies (with fallback)
  static async getHistoricalInterest(userId?: string, limit: number = 20): Promise<MovieData[]> {
    if (userId) {
      try {
        console.log('🔄 Attempting to fetch real historical interest movies...');
        const response = await MovieGenieAPI.getUserHistoricalInterest(parseInt(userId), limit);
        const movies = response.movies?.map(this.transformApiMovie) || [];
        console.log('✅ Got real historical interest movies:', movies.length);
        return movies;
      } catch (error) {
        console.warn('⚠️ Real historical interest failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock historical interest movies');
    return Array.from({length: limit}, (_, i) =>
      this.createMockMovie(i + 300, `Historical Interest ${i + 1}`, 'Historical')
    );
  }

  // Get user's watched movies (with fallback)
  static async getUserWatchedMovies(userId?: string, limit: number = 20): Promise<MovieData[]> {
    if (userId) {
      try {
        console.log('🔄 Attempting to fetch real watched movies...');
        const response = await MovieGenieAPI.getUserWatchedMovies(parseInt(userId), limit);
        const movies = response.movies?.map((movie: any) => ({
          ...this.transformApiMovie(movie),
          watched: true  // Preserve watched flag from backend
        })) || [];
        console.log('✅ Got real watched movies:', movies.length);
        return movies;
      } catch (error) {
        console.warn('⚠️ Real watched movies failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock watched movies for user:', userId);
    return Array.from({length: limit}, (_, i) =>
      this.createMockMovie(i + 400, `Watched Movie ${i + 1}`, 'Watched')
    ).map(movie => ({ ...movie, watched: true }));
  }

  // Check if a data source is using real data
  static isUsingRealData(source: keyof typeof DATA_SOURCE_CONFIG): boolean {
    return DATA_SOURCE_CONFIG[source];
  }

  // Get configuration status
  static getDataSourceStatus() {
    return {
      popular: DATA_SOURCE_CONFIG.popular ? '🌐 Real API' : '📝 Mock Data',
      search: DATA_SOURCE_CONFIG.search ? '🌐 Real API' : '📝 Mock Data',
      recommendations: DATA_SOURCE_CONFIG.recommendations ? '🌐 Real API' : '📝 Mock Data',
      movieDetails: DATA_SOURCE_CONFIG.movieDetails ? '🌐 Real API' : '📝 Mock Data',
    };
  }
}

export default MovieDataService;