use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path as AxumPath},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use futures_util::StreamExt;
use std::net::SocketAddr;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tokio_util::io::StreamReader;
use tokio_tar::Archive;
use nra_core::stream_writer::StreamBetaWriter;
use std::fs;

/// Maximum upload size: 10 GiB
const MAX_UPLOAD_SIZE: usize = 10 * 1024 * 1024 * 1024;

async fn upload_handler(
    AxumPath(name): AxumPath<String>,
    body: Body,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!("Received upload request for archive: {}", name);
    // CRIT-3 fix: Validate archive name to prevent path traversal
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains('\0')
    {
        return Err((StatusCode::BAD_REQUEST, "Invalid archive name".into()));
    }

    let stream = body.into_data_stream().map(|res| {
        res.map_err(|e| std::io::Error::other(e.to_string()))
    });
    let stream_reader = StreamReader::new(stream);
    
    let mut archive = Archive::new(stream_reader);
    let mut entries = archive.entries().map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let out_path = format!("nra-registry-server/data/{}.nra", name);
    let _ = fs::create_dir_all("nra-registry-server/data");

    let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, Vec<u8>)>(10);
    
    let writer_task = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let file = std::fs::File::create(&out_path)?;
        let mut writer = StreamBetaWriter::new(file)?;
        while let Some((id, data)) = rx.blocking_recv() {
            writer.add_file(&id, &data)?;
        }
        writer.finalize()?;
        Ok(())
    });

    let mut count = 0;
    while let Some(entry_res) = entries.next().await {
        let mut entry = entry_res.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let header = entry.header();
        if header.entry_type().is_file() {
            let path = entry.path().map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let path_str = path.to_string_lossy().to_string();
            
            use tokio::io::AsyncReadExt;
            let mut data = Vec::new();
            entry.read_to_end(&mut data).await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            
            if tx.send((path_str, data)).await.is_err() {
                return Err((StatusCode::INTERNAL_SERVER_ERROR, "Writer task panicked".to_string()));
            }
            count += 1;
        }
    }

    drop(tx);
    // WARN-1 fix: handle JoinError gracefully instead of unwrap()
    writer_task.await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Writer task panicked: {e}")))?  
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::OK, format!("Successfully packaged {} files into {}.nra", count, name)))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nra_registry_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let serve_dir = ServeDir::new("nra-registry-server/data");

    let app = Router::new()
        .route("/api/v1/upload/:name", post(upload_handler))
        .nest_service("/archives", serve_dir)
        .layer(TraceLayer::new_for_http())
        // CRIT-2 fix: limit upload body size
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_SIZE));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("NRA Registry Server listening on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
