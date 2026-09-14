"""Domain errors translated to stable HTTP responses by the app layer."""


class RetailAPIError(Exception):
    def __init__(self, message: str, *, status_code: int = 400,
                 code: str = "invalid_request"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
