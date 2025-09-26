import { useState } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";

interface MovieSearchProps {
  onSearch: (query: string) => void;
}

export function MovieSearch({ onSearch }: MovieSearchProps) {
  const [query, setQuery] = useState("");

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onSearch(query);
    }
  };

  return (
    <div className="relative mb-6">
      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
      <Input
        type="search"
        placeholder="Search movies..."
        className="pl-10 bg-input border-border focus:ring-2 focus:ring-primary/50 transition-all duration-200"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
      />
    </div>
  );
}