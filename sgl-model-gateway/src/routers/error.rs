use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub fn internal_error(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        message,
        None,
        None,
    )
}

pub fn bad_request(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        None,
        None,
    )
}

pub fn bad_request_with_code(message: impl Into<String>, code: impl Into<String>) -> Response {
    create_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        None,
        Some(code.into()),
    )
}

pub fn bad_request_with_param(
    message: impl Into<String>,
    param: impl Into<String>,
    code: Option<impl Into<String>>,
) -> Response {
    create_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        Some(param.into()),
        code.map(|c| c.into()),
    )
}

pub fn not_found(message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, "invalid_request_error", message, None, None)
}

pub fn not_found_with_code(message: impl Into<String>, code: impl Into<String>) -> Response {
    create_error(
        StatusCode::NOT_FOUND,
        "not_found_error",
        message,
        None,
        Some(code.into()),
    )
}

pub fn service_unavailable(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable",
        message,
        None,
        None,
    )
}

pub fn service_unavailable_with_param(
    message: impl Into<String>,
    param: impl Into<String>,
    code: impl Into<String>,
) -> Response {
    create_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable",
        message,
        Some(param.into()),
        Some(code.into()),
    )
}

pub fn failed_dependency(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::FAILED_DEPENDENCY,
        "external_connector_error",
        message,
        None,
        None,
    )
}

pub fn not_implemented(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented_error",
        message,
        None,
        None,
    )
}

pub fn internal_error_with_code(message: impl Into<String>, code: impl Into<String>) -> Response {
    create_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        message,
        None,
        Some(code.into()),
    )
}

fn create_error(
    status_code: StatusCode,
    error_type: &str,
    message: impl Into<String>,
    param: Option<String>,
    code: Option<String>,
) -> Response {
    let msg = message.into();
    let mut error_obj = json!({
        "message": msg,
        "type": error_type,
    });

    // Use custom code if provided, otherwise use status code number
    if let Some(code_str) = code {
        error_obj["code"] = json!(code_str);
    } else {
        error_obj["code"] = json!(status_code.as_u16());
    }

    // Add param if provided
    if let Some(param_str) = param {
        error_obj["param"] = json!(param_str);
    }

    (
        status_code,
        Json(json!({
            "error": error_obj
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_error_string() {
        let response = internal_error("Test error");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_internal_error_format() {
        let response = internal_error(format!("Error: {}", 42));
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_bad_request() {
        let response = bad_request("Invalid input");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_not_found() {
        let response = not_found("Resource not found");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_service_unavailable() {
        let response = service_unavailable("No workers");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
