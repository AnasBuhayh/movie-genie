import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ThumbsUp, ThumbsDown } from "lucide-react";

interface MovieDetailsProps {
  movie: {
    title: string;
    genres: string[];
    description: string;
    likes: string[];
    dislikes: string[];
  };
}

export function MovieDetails({ movie }: MovieDetailsProps) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-4 text-foreground">{movie.title}</h1>
        <Separator className="border-border" />
      </div>

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

      <div>
        <p className="text-foreground leading-relaxed text-sm">
          {movie.description}
        </p>
        <Separator className="border-border mt-4" />
      </div>

      <Card className="p-4 bg-gradient-to-br from-card to-card/80 shadow-lg">
        <div className="space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <ThumbsUp className="h-4 w-4 text-rating-positive" />
              <h3 className="text-lg font-semibold text-rating-positive">People Like</h3>
            </div>
            <ul className="space-y-2">
              {movie.likes.map((like, index) => (
                <li key={index} className="text-sm text-muted-foreground border-l-2 border-rating-positive/30 pl-3">
                  {like}
                </li>
              ))}
            </ul>
          </div>

          <Separator className="border-border/50" />

          <div>
            <div className="flex items-center gap-2 mb-3">
              <ThumbsDown className="h-4 w-4 text-rating-negative" />
              <h3 className="text-lg font-semibold text-rating-negative">People Don't Like</h3>
            </div>
            <ul className="space-y-2">
              {movie.dislikes.map((dislike, index) => (
                <li key={index} className="text-sm text-muted-foreground border-l-2 border-rating-negative/30 pl-3">
                  {dislike}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}