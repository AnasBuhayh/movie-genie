"""
API Response Utilities for consistent response formatting
"""

from typing import Any, Dict, Optional
from flask import jsonify

class APIResponse:
    """Standardized API response format"""

    @staticmethod
    def success(data: Any = None,
                message: str = "Success",
                status_code: int = 200) -> tuple:
        """
        Format successful API response

        Args:
            data: Response data
            message: Success message
            status_code: HTTP status code

        Returns:
            Tuple of (response, status_code)
        """
        response = {
            'success': True,
            'message': message,
            'data': data
        }

        return jsonify(response), status_code

    @staticmethod
    def error(message: str = "An error occurred",
              error_code: Optional[str] = None,
              details: Any = None,
              status_code: int = 400) -> tuple:
        """
        Format error API response

        Args:
            message: Error message
            error_code: Error code for client handling
            details: Additional error details
            status_code: HTTP status code

        Returns:
            Tuple of (response, status_code)
        """
        response = {
            'success': False,
            'message': message,
            'error': {
                'code': error_code,
                'details': details
            }
        }

        return jsonify(response), status_code

    @staticmethod
    def paginated(data: list,
                  total: int,
                  page: int = 1,
                  per_page: int = 20,
                  message: str = "Success") -> tuple:
        """
        Format paginated API response

        Args:
            data: Page data
            total: Total number of items
            page: Current page number
            per_page: Items per page
            message: Success message

        Returns:
            Tuple of (response, status_code)
        """
        response = {
            'success': True,
            'message': message,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page  # Ceiling division
            }
        }

        return jsonify(response), 200

def format_success_response(data: Any = None,
                           message: str = "Success",
                           status_code: int = 200) -> tuple:
    """Shorthand for success response"""
    return APIResponse.success(data, message, status_code)

def format_error_response(message: str = "An error occurred",
                         error_code: Optional[str] = None,
                         details: Any = None,
                         status_code: int = 400) -> tuple:
    """Shorthand for error response"""
    return APIResponse.error(message, error_code, details, status_code)