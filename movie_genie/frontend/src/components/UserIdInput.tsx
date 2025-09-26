import React, { useState, useEffect } from 'react';
import { User } from "lucide-react";
import { Input } from "@/components/ui/input";
import { MovieGenieAPI, UserInfo } from '../lib/api';

interface UserIdInputProps {
  value: string;
  onChange: (userId: string) => void;
  onValidChange?: (isValid: boolean) => void;
}

export const UserIdInput: React.FC<UserIdInputProps> = ({
  value,
  onChange,
  onValidChange
}) => {
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        setIsLoading(true);
        const info = await MovieGenieAPI.getUserInfo();
        setUserInfo(info);
        setError(null);
      } catch (err) {
        setError('Failed to load user information');
        console.error('Failed to fetch user info:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserInfo();
  }, []);

  const isValidUserId = (id: string): boolean => {
    if (!userInfo || !userInfo.user_id_range || !id.trim()) return false;

    const numId = parseInt(id);
    if (isNaN(numId)) return false;

    return numId >= userInfo.user_id_range.min && numId <= userInfo.user_id_range.max;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    onChange(newValue);

    if (onValidChange) {
      onValidChange(isValidUserId(newValue));
    }
  };

  const getValidationMessage = (): string | null => {
    if (!value.trim()) return null;
    if (!userInfo || !userInfo.user_id_range) return null;

    const numId = parseInt(value);
    if (isNaN(numId)) {
      return 'Please enter a valid number';
    }

    if (numId < userInfo.user_id_range.min || numId > userInfo.user_id_range.max) {
      return `User ID must be between ${userInfo.user_id_range.min} and ${userInfo.user_id_range.max}`;
    }

    return null;
  };

  const validationMessage = getValidationMessage();
  const isValid = !validationMessage && value.trim() !== '';

  return (
    <div className="relative mb-6">
      <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
      <Input
        type="text"
        value={value}
        onChange={handleInputChange}
        placeholder={userInfo && userInfo.user_id_range ? `Enter user ID (1-${userInfo.user_id_range.max})` : "Enter user ID..."}
        className="pl-10 bg-input border-border focus:ring-2 focus:ring-primary/50 transition-all duration-200"
        disabled={isLoading}
      />
    </div>
  );
};

export default UserIdInput;