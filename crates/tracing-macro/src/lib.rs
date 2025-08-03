#[macro_export]
macro_rules! scope {
    ($name:expr, $expr:expr) => {{
        let span = tracing::span!(tracing::Level::INFO, $name);
        let _guard = span.enter();
        $expr
    }};
}
