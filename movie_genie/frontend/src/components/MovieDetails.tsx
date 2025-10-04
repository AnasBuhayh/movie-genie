import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ThumbsUp, ThumbsDown, Star, Users, Clock } from "lucide-react";

interface MovieDetailsProps {
  movie: {
    title: string;
    genres: string[];
    description: string;
    rating?: number;
    voteCount?: number;
    runtime?: number;
    likes: string[];
    dislikes: string[];
  };
}

export function MovieDetails({ movie }: MovieDetailsProps) {
  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h1 className="text-3xl font-bold mb-4 text-foreground">{movie.title}</h1>
        <Separator className="border-border" />
      </div>

      {/* Genres */}
      <div>
        <div className="flex flex-wrap gap-2 mb-4">
          {movie.genres.map((genre) => (
            <Badge key={genre} variant="secondary" className="bg-secondary/80">
              {genre}
            </Badge>
          ))}
        </div>
        <Separator className="border-border" />
      </div>

      {/* Description */}
      <div>
        <p className="text-foreground leading-relaxed text-sm">
          {movie.description}
        </p>
        <Separator className="border-border mt-4" />
      </div>

      {/* Movie Stats Section */}
      {(movie.rating || movie.voteCount || movie.runtime) && (
        <Card className="p-4 bg-gradient-to-br from-primary/5 to-accent/5">
          <div className="grid grid-cols-3 gap-4">
            {movie.rating && (
              <div className="flex items-center gap-2">
                <Star className="h-5 w-5 text-yellow-500 fill-yellow-500" />
                <div>
                  <p className="text-sm font-semibold text-foreground">{movie.rating.toFixed(1)}/10</p>
                  <p className="text-xs text-muted-foreground">Rating</p>
                </div>
              </div>
            )}

            {movie.voteCount && (
              <div className="flex items-center gap-2">
                <Users className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-sm font-semibold text-foreground">{movie.voteCount.toLocaleString()}</p>
                  <p className="text-xs text-muted-foreground">Votes</p>
                </div>
              </div>
            )}

            {movie.runtime && (
              <div className="flex items-center gap-2">
                <Clock className="h-5 w-5 text-purple-500" />
                <div>
                  <p className="text-sm font-semibold text-foreground">{movie.runtime} min</p>
                  <p className="text-xs text-muted-foreground">Runtime</p>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* People Like / Don't Like Section */}
      <Card className="p-4 bg-gradient-to-br from-card to-card/80 shadow-lg">
        <div className="space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <ThumbsUp className="h-4 w-4 text-rating-positive" />
              <h3 className="text-lg font-semibold text-rating-positive">People Like</h3>
            </div>
            {movie.likes.length > 0 ? (
              <ul className="space-y-2">
                {movie.likes.map((like, index) => (
                  <li key={index} className="text-sm text-muted-foreground border-l-2 border-rating-positive/30 pl-3">
                    {like}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground italic">Coming soon...</p>
            )}
          </div>

          <Separator className="border-border/50" />

          <div>
            <div className="flex items-center gap-2 mb-3">
              <ThumbsDown className="h-4 w-4 text-rating-negative" />
              <h3 className="text-lg font-semibold text-rating-negative">People Don't Like</h3>
            </div>
            {movie.dislikes.length > 0 ? (
              <ul className="space-y-2">
                {movie.dislikes.map((dislike, index) => (
                  <li key={index} className="text-sm text-muted-foreground border-l-2 border-rating-negative/30 pl-3">
                    {dislike}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground italic">Coming soon...</p>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}