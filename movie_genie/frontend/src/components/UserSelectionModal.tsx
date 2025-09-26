import React, { useState, useEffect } from 'react';
import { User, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { MovieGenieAPI, UserInfo } from '../lib/api';

interface UserSelectionModalProps {
  isOpen: boolean;
  onUserSelect: (userId: string) => void;
}

export const UserSelectionModal: React.FC<UserSelectionModalProps> = ({
  isOpen,
  onUserSelect
}) => {
  const [userId, setUserId] = useState("");
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchUserInfo();
    }
  }, [isOpen]);

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

  const isValidUserId = (id: string): boolean => {
    if (!userInfo || !userInfo.user_id_range || !id.trim()) return false;
    const numId = parseInt(id);
    if (isNaN(numId)) return false;
    return numId >= userInfo.user_id_range.min && numId <= userInfo.user_id_range.max;
  };

  const handleSubmit = async (e: React.FormEvent | React.KeyboardEvent) => {
    e.preventDefault();

    if (!canSubmit || isLoggingIn) return;

    setIsLoggingIn(true);

    // Simulate login/loading process
    await new Promise(resolve => setTimeout(resolve, 1500));

    setIsLoggingIn(false);
    onUserSelect(userId);
  };

  const getValidationMessage = (): string | null => {
    if (!userId.trim()) return null;
    if (!userInfo || !userInfo.user_id_range) return null;

    const numId = parseInt(userId);
    if (isNaN(numId)) {
      return 'Please enter a valid number';
    }

    if (numId < userInfo.user_id_range.min || numId > userInfo.user_id_range.max) {
      return `User ID must be between ${userInfo.user_id_range.min} and ${userInfo.user_id_range.max}`;
    }

    return null;
  };

  const validationMessage = getValidationMessage();
  const canSubmit = isValidUserId(userId);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4 shadow-xl">
        <div className="text-center mb-6">
          <User className="h-16 w-16 mx-auto text-primary mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Welcome to Movie Genie
          </h2>
          <p className="text-gray-600">
            Please select a user to get personalized movie recommendations
          </p>
        </div>

        {isLoading ? (
          <div className="text-center py-8">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
            <p className="text-gray-600">Loading user information...</p>
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <p className="text-red-600 mb-4">{error}</p>
            <Button onClick={fetchUserInfo} variant="outline">
              Try Again
            </Button>
          </div>
        ) : isLoggingIn ? (
          <div className="text-center py-8">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
            <p className="text-gray-600">Logging in as User {userId}...</p>
            <p className="text-sm text-gray-500 mt-2">Loading your personalized experience</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="modal-user-id" className="block text-sm font-medium text-gray-700 mb-2">
                User ID
              </label>

              {userInfo && userInfo.user_id_range && (
                <div className="mb-3 p-3 bg-blue-50 rounded-md">
                  <p className="text-sm text-blue-800">
                    <strong>Valid Range:</strong> {userInfo.user_id_range.min} - {userInfo.user_id_range.max}
                  </p>
                  <p className="text-xs text-blue-600 mt-1">
                    Total users: {userInfo.user_id_range.total.toLocaleString()}
                  </p>
                </div>
              )}

              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                <Input
                  id="modal-user-id"
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && canSubmit) {
                      handleSubmit(e as any);
                    }
                  }}
                  placeholder="Enter your user ID..."
                  className="pl-10"
                  autoFocus
                />
              </div>

              {validationMessage && (
                <p className="mt-2 text-sm text-red-600">{validationMessage}</p>
              )}
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={!canSubmit || isLoggingIn}
              onClick={() => console.log(`User ID submitted: ${userId}`)}
            >
              Continue as User {userId || '...'}
            </Button>
          </form>
        )}
      </div>
    </div>
  );
};

export default UserSelectionModal;