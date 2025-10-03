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
const DATA_SOURCE_CONFIG = {
  popular: process.env.VITE_USE_REAL_POPULAR === 'true',
  search: process.env.VITE_USE_REAL_SEARCH === 'true',
  recommendations: process.env.VITE_USE_REAL_RECOMMENDATIONS === 'true',
  movieDetails: process.env.VITE_USE_REAL_MOVIE_DETAILS === 'true',
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
        const response = await MovieGenieAPI.getPersonalizedRecommendations(userId);
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

  // Search movies (with fallback)
  static async searchMovies(query: string, limit: number = 20): Promise<SearchResults> {
    if (DATA_SOURCE_CONFIG.search && query.trim()) {
      try {
        console.log('🔄 Attempting real semantic search for:', query);
        const response = await MovieGenieAPI.searchMovies(query, true);
        const movies = response.movies?.map(this.transformApiMovie) || [];
        console.log('✅ Got real search results:', movies.length);

        return {
          movies: movies.slice(0, limit),
          total: movies.length,
          query,
          hasRealData: true,
        };
      } catch (error) {
        console.warn('⚠️ Real search failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock search results for:', query);
    const mockMovies = Array.from({length: limit}, (_, i) =>
      this.createMockMovie(i + 200, `${query} Result ${i + 1}`, 'Search')
    );

    return {
      movies: mockMovies,
      total: mockMovies.length,
      query,
      hasRealData: false,
    };
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

  // Get historical interest movies (mock for now)
  static async getHistoricalInterest(userId?: string, limit: number = 20): Promise<MovieData[]> {
    console.log('📝 Using mock historical interest movies');
    return Array.from({length: limit}, (_, i) =>
      this.createMockMovie(i + 300, `Historical Interest ${i + 1}`, 'Historical')
    );
  }

  // Get user's watched movies (mock for now)
  static async getUserWatchedMovies(userId?: string, limit: number = 20): Promise<MovieData[]> {
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