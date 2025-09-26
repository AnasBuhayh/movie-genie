"""
Custom API Errors for Movie Genie Backend
"""

class APIError(Exception):
    """Base API error class"""

    def __init__(self,
                 message: str,
                 status_code: int = 400,
                 error_code: str = None,
                 details: any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details

class ValidationError(APIError):
    """Input validation error"""

    def __init__(self, message: str, details: any = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )

class NotFoundError(APIError):
    """Resource not found error"""

    def __init__(self, resource: str, resource_id: any = None):
        message = f"{resource} not found"
        if resource_id:
            message += f" (ID: {resource_id})"

        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={'resource': resource, 'id': resource_id}
        )

class ServiceUnavailableError(APIError):
    """Service unavailable error (e.g., ML models not loaded)"""

    def __init__(self, service_name: str):
        super().__init__(
            message=f"{service_name} is currently unavailable",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details={'service': service_name}
        )

class InternalServerError(APIError):
    """Internal server error"""

    def __init__(self, message: str = "Internal server error", details: any = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            details=details
        )