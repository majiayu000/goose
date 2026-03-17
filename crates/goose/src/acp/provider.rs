use agent_client_protocol_schema::AGENT_METHOD_NAMES;
use anyhow::{Context, Result};
use async_stream::try_stream;
use rmcp::model::{Role, Tool};
use sacp::schema::{
    AuthMethod, CloseSessionRequest, ContentBlock, ContentChunk, EnvVariable, HttpHeader,
    ImageContent, InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
    McpCapabilities, McpServer, McpServerHttp, McpServerStdio, NewSessionRequest,
    NewSessionResponse, PromptRequest, PromptResponse, ProtocolVersion, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SessionConfigKind,
    SessionConfigOptionCategory, SessionId, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionModeRequest, SetSessionModeResponse,
    SetSessionModelRequest, StopReason, TextContent, ToolCallContent,
};
use sacp::{Agent, Client, ConnectionTo};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::process::Stdio;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex as TokioMutex};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::acp::{map_permission_response, PermissionDecision, PermissionMapping};
use crate::config::{ExtensionConfig, GooseMode};
use crate::conversation::message::{Message, MessageContent};
use crate::model::ModelConfig;
use crate::permission::permission_confirmation::PrincipalType;
use crate::permission::{Permission, PermissionConfirmation};
use crate::providers::base::{MessageStream, PermissionRouting, Provider};
use crate::providers::errors::ProviderError;

pub struct AcpProviderConfig {
    pub command: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub env_remove: Vec<String>,
    pub work_dir: PathBuf,
    pub mcp_servers: Vec<McpServer>,
    pub session_mode_id: Option<String>,
    pub mode_mapping: HashMap<GooseMode, String>,
    pub permission_mapping: PermissionMapping,
    pub notification_callback: Option<Arc<dyn Fn(SessionNotification) + Send + Sync>>,
}

enum ClientRequest {
    NewSession {
        response_tx: oneshot::Sender<Result<NewSessionResponse>>,
    },
    ListSessions {
        response_tx: oneshot::Sender<Result<ListSessionsResponse>>,
    },
    SetMode {
        session_id: SessionId,
        mode_id: String,
        response_tx: oneshot::Sender<Result<()>>,
    },
    SetModel {
        session_id: SessionId,
        model_id: String,
        response_tx: oneshot::Sender<Result<()>>,
    },
    SetConfigOption {
        session_id: SessionId,
        config_id: String,
        value: String,
        response_tx: oneshot::Sender<Result<()>>,
    },
    Prompt {
        session_id: SessionId,
        content: Vec<ContentBlock>,
        response_tx: mpsc::Sender<AcpUpdate>,
    },
    // For ACP methods not yet in agent-client-protocol-schema (e.g. session/delete)
    Untyped {
        method: String,
        params: serde_json::Value,
        response_tx: oneshot::Sender<Result<serde_json::Value>>,
    },
}

#[derive(Debug)]
enum AcpUpdate {
    Text(String),
    Thought(String),
    ToolCallStart {
        id: String,
    },
    ToolCallComplete {
        id: String,
    },
    PermissionRequest {
        request: Box<RequestPermissionRequest>,
        response_tx: oneshot::Sender<RequestPermissionResponse>,
    },
    Complete(StopReason),
    Error(String),
}

pub struct AcpProvider {
    name: String,
    model: ModelConfig,
    goose_mode: Arc<Mutex<GooseMode>>,
    tx: Option<mpsc::Sender<ClientRequest>>,
    loop_thread: Option<std::thread::JoinHandle<()>>,
    mode_mapping: HashMap<GooseMode, String>,
    permission_mapping: PermissionMapping,
    rejected_tool_calls: Arc<TokioMutex<HashSet<String>>>,
    pending_confirmations:
        Arc<TokioMutex<HashMap<String, oneshot::Sender<PermissionConfirmation>>>>,
    goose_to_acp_id: Arc<TokioMutex<HashMap<String, NewSessionResponse>>>,
    acp_to_goose_id: Arc<TokioMutex<HashMap<String, String>>>,
    /// Per-session model tracking for detecting model changes in stream().
    session_model: Arc<TokioMutex<HashMap<String, String>>>,
    auth_methods: Vec<AuthMethod>,
}

impl std::fmt::Debug for AcpProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcpProvider")
            .field("name", &self.name)
            .field("model", &self.model)
            .finish()
    }
}

// Dedicated runtime on an OS thread so session/close completes even during
// main runtime shutdown. See reqwest InnerClientHandle.
fn spawn_client_loop(
    fut: impl Future<Output = ()> + Send + 'static,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build ACP client runtime");
        rt.block_on(fut)
    })
}

impl AcpProvider {
    pub async fn connect(
        name: String,
        model: ModelConfig,
        goose_mode: GooseMode,
        config: AcpProviderConfig,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel(32);
        let (init_tx, init_rx) = oneshot::channel();
        let mode_mapping = config.mode_mapping.clone();
        let permission_mapping = config.permission_mapping.clone();
        let rejected_tool_calls = Arc::new(TokioMutex::new(HashSet::new()));
        let goose_mode = Arc::new(Mutex::new(goose_mode));

        let client_loop = AcpClientLoop::new(config, goose_mode.clone());
        let loop_thread = spawn_client_loop(client_loop.spawn(rx, init_tx));

        let init_response = init_rx
            .await
            .context("ACP client initialization cancelled")??;

        Ok(Self::new_with_runtime(
            name,
            model,
            goose_mode,
            tx,
            loop_thread,
            mode_mapping,
            permission_mapping,
            rejected_tool_calls,
            init_response.auth_methods,
        ))
    }

    pub async fn connect_with_transport<R, W>(
        name: String,
        model: ModelConfig,
        goose_mode: GooseMode,
        config: AcpProviderConfig,
        read: R,
        write: W,
    ) -> Result<Self>
    where
        R: futures::AsyncRead + Unpin + Send + 'static,
        W: futures::AsyncWrite + Unpin + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel(32);
        let (init_tx, init_rx) = oneshot::channel();
        let mode_mapping = config.mode_mapping.clone();
        let permission_mapping = config.permission_mapping.clone();
        let rejected_tool_calls = Arc::new(TokioMutex::new(HashSet::new()));
        let goose_mode = Arc::new(Mutex::new(goose_mode));
        let transport = sacp::ByteStreams::new(write, read);
        let client_loop = AcpClientLoop::new(config, goose_mode.clone());
        let loop_thread = spawn_client_loop(async move {
            if let Err(e) = client_loop.run(transport, &mut rx, init_tx).await {
                tracing::error!("ACP protocol error: {e}");
            }
        });

        let init_response = init_rx
            .await
            .context("ACP client initialization cancelled")??;

        Ok(Self::new_with_runtime(
            name,
            model,
            goose_mode,
            tx,
            loop_thread,
            mode_mapping,
            permission_mapping,
            rejected_tool_calls,
            init_response.auth_methods,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_runtime(
        name: String,
        model: ModelConfig,
        goose_mode: Arc<Mutex<GooseMode>>,
        tx: mpsc::Sender<ClientRequest>,
        loop_thread: std::thread::JoinHandle<()>,
        mode_mapping: HashMap<GooseMode, String>,
        permission_mapping: PermissionMapping,
        rejected_tool_calls: Arc<TokioMutex<HashSet<String>>>,
        auth_methods: Vec<AuthMethod>,
    ) -> Self {
        Self {
            name,
            model,
            goose_mode,
            tx: Some(tx),
            loop_thread: Some(loop_thread),
            mode_mapping,
            permission_mapping,
            rejected_tool_calls,
            pending_confirmations: Arc::new(TokioMutex::new(HashMap::new())),
            goose_to_acp_id: Arc::new(TokioMutex::new(HashMap::new())),
            acp_to_goose_id: Arc::new(TokioMutex::new(HashMap::new())),
            session_model: Arc::new(TokioMutex::new(HashMap::new())),
            auth_methods,
        }
    }

    pub fn auth_methods(&self) -> &[AuthMethod] {
        &self.auth_methods
    }

    pub async fn new_session(&self) -> Result<NewSessionResponse> {
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::NewSession { response_tx })
            .await
            .context("ACP client is unavailable")?;
        response_rx
            .await
            .context(format!("ACP {} cancelled", AGENT_METHOD_NAMES.session_new))?
    }

    pub async fn list_sessions(&self) -> Result<ListSessionsResponse> {
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::ListSessions { response_tx })
            .await
            .context("ACP client is unavailable")?;
        let raw = response_rx.await.context("ACP request cancelled")??;
        let acp_to_goose = self.acp_to_goose_id.lock().await;
        Ok(map_sessions_to_goose_ids(raw, &acp_to_goose))
    }

    async fn resolve_acp_session_id(&self, goose_id: &str) -> Result<SessionId> {
        let map = self.goose_to_acp_id.lock().await;
        map.get(goose_id)
            .map(|r| r.session_id.clone())
            .ok_or_else(|| {
                sacp::Error::invalid_params()
                    .data(format!("Session not found: {goose_id}"))
                    .into()
            })
    }

    pub(crate) async fn send_set_mode(&self, goose_id: &str, mode_id: String) -> Result<()> {
        let session_id = self.resolve_acp_session_id(goose_id).await?;
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::SetMode {
                session_id,
                mode_id,
                response_tx,
            })
            .await
            .context("ACP client is unavailable")?;
        response_rx.await.context("ACP request cancelled")?
    }

    pub(crate) async fn send_set_model(&self, goose_id: &str, model_id: String) -> Result<()> {
        let session_id = self.resolve_acp_session_id(goose_id).await?;
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::SetModel {
                session_id,
                model_id,
                response_tx,
            })
            .await
            .context("ACP client is unavailable")?;
        response_rx.await.context("ACP request cancelled")?
    }

    pub(crate) async fn send_set_config_option(
        &self,
        goose_id: &str,
        config_id: String,
        value: String,
    ) -> Result<()> {
        let session_id = self.resolve_acp_session_id(goose_id).await?;
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::SetConfigOption {
                session_id,
                config_id,
                value,
                response_tx,
            })
            .await
            .context("ACP client is unavailable")?;
        response_rx.await.context("ACP request cancelled")?
    }

    pub async fn delete_session(&self, goose_id: &str) -> Result<()> {
        let session_id = self.resolve_acp_session_id(goose_id).await?;
        self.send_untyped(
            "session/delete",
            serde_json::json!({ "session_id": session_id.0 }),
        )
        .await
        .map(|_| ())
    }

    pub async fn send_untyped(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::Untyped {
                method: method.to_string(),
                params,
                response_tx,
            })
            .await
            .context("ACP client is unavailable")?;
        response_rx.await.context("ACP request cancelled")?
    }

    /// Returns true when the given goose session ID has been established.
    pub async fn has_session(&self, goose_id: &str) -> bool {
        self.goose_to_acp_id.lock().await.contains_key(goose_id)
    }

    /// Checks whether the downstream agent advertised a config_option with the
    /// given category for this session. Used to decide between set_config_option
    /// (preferred when available) and legacy methods (set_mode / set_model).
    async fn session_has_config_option(
        &self,
        goose_id: &str,
        category: SessionConfigOptionCategory,
    ) -> bool {
        let map = self.goose_to_acp_id.lock().await;
        map.get(goose_id)
            .and_then(|r| r.config_options.as_ref())
            .is_some_and(|opts| opts.iter().any(|o| o.category.as_ref() == Some(&category)))
    }

    pub async fn handle_permission_confirmation(
        &self,
        request_id: &str,
        confirmation: &PermissionConfirmation,
    ) -> bool {
        let mut pending = self.pending_confirmations.lock().await;
        if let Some(tx) = pending.remove(request_id) {
            let _ = tx.send(confirmation.clone());
            return true;
        }
        false
    }

    pub async fn ensure_session(
        &self,
        session_id: Option<&str>,
    ) -> Result<NewSessionResponse, ProviderError> {
        if let Some(session_id) = session_id {
            if let Some(response) = self.goose_to_acp_id.lock().await.get(session_id) {
                return Ok(response.clone());
            }
        }

        let response = self.new_session().await.map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to create ACP session: {e}"))
        })?;

        if let Some(session_id) = session_id {
            self.goose_to_acp_id
                .lock()
                .await
                .insert(session_id.to_string(), response.clone());
            self.acp_to_goose_id
                .lock()
                .await
                .insert(response.session_id.0.to_string(), session_id.to_string());
            // Initialize model tracking so stream() can detect changes.
            self.session_model
                .lock()
                .await
                .entry(session_id.to_string())
                .or_insert_with(|| self.model.model_name.clone());
        }

        Ok(response)
    }

    async fn prompt(
        &self,
        session_id: SessionId,
        content: Vec<ContentBlock>,
    ) -> Result<mpsc::Receiver<AcpUpdate>> {
        let (response_tx, response_rx) = mpsc::channel(64);
        self.tx
            .as_ref()
            .unwrap()
            .send(ClientRequest::Prompt {
                session_id,
                content,
                response_tx,
            })
            .await
            .context("ACP client is unavailable")?;
        Ok(response_rx)
    }
}

#[async_trait::async_trait]
impl Provider for AcpProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn update_mode(&self, session_id: &str, mode: GooseMode) -> Result<(), ProviderError> {
        let map = self.goose_to_acp_id.lock().await;
        if map.is_empty() {
            // Pre-initialization: no ACP session yet, just store the mode.
            // The shared Arc<Mutex<GooseMode>> is read at session creation time.
            drop(map);
        } else {
            drop(map);
            let mode_str = self.mode_mapping[&mode].clone();
            // Prefer set_config_option when the agent advertises config_options (spec
            // recommendation). Fall back to legacy set_mode for agents that don't.
            if self
                .session_has_config_option(session_id, SessionConfigOptionCategory::Mode)
                .await
            {
                self.send_set_config_option(session_id, "mode".into(), mode_str)
                    .await
                    .map_err(|e| {
                        ProviderError::RequestFailed(format!("Failed to set mode: {e}"))
                    })?;
            } else {
                self.send_set_mode(session_id, mode_str)
                    .await
                    .map_err(|e| {
                        ProviderError::RequestFailed(format!("Failed to set mode: {e}"))
                    })?;
            }
        }

        let mut current = self
            .goose_mode
            .lock()
            .map_err(|_| ProviderError::RequestFailed("Failed to update mode".into()))?;
        *current = mode;
        Ok(())
    }

    fn permission_routing(&self) -> PermissionRouting {
        PermissionRouting::ActionRequired
    }

    async fn handle_permission_confirmation(
        &self,
        request_id: &str,
        confirmation: &PermissionConfirmation,
    ) -> bool {
        AcpProvider::handle_permission_confirmation(self, request_id, confirmation).await
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        _system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let response = self.ensure_session(Some(session_id)).await?;

        // Model forwarding: Provider trait has no update_model method — model is
        // implicitly passed to stream(), so mismatch detection here is the
        // production path for forwarding model changes to the downstream agent.
        {
            let new_model = &model_config.model_name;
            let tracked = self.session_model.lock().await.get(session_id).cloned();
            if tracked.as_deref() != Some(new_model) {
                // Model changed since session creation or last stream() call:
                // forward to downstream. Prefer set_config_option when the agent
                // advertises it (spec recommendation), fall back to legacy set_model.
                if self
                    .session_has_config_option(session_id, SessionConfigOptionCategory::Model)
                    .await
                {
                    self.send_set_config_option(session_id, "model".into(), new_model.clone())
                        .await
                        .map_err(|e| {
                            ProviderError::RequestFailed(format!("Failed to set model: {e}"))
                        })?;
                } else {
                    self.send_set_model(session_id, new_model.clone())
                        .await
                        .map_err(|e| {
                            ProviderError::RequestFailed(format!("Failed to set model: {e}"))
                        })?;
                }
                self.session_model
                    .lock()
                    .await
                    .insert(session_id.to_string(), new_model.clone());
            }
        }

        let prompt_blocks = messages_to_prompt(messages);
        let mut rx = self
            .prompt(response.session_id, prompt_blocks)
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to send ACP prompt: {e}")))?;

        let pending_confirmations = self.pending_confirmations.clone();
        let rejected_tool_calls = self.rejected_tool_calls.clone();
        let permission_mapping = self.permission_mapping.clone();
        let goose_mode = *self
            .goose_mode
            .lock()
            .map_err(|_| ProviderError::RequestFailed("goose_mode lock poisoned".into()))?;

        let reject_all_tools = goose_mode == GooseMode::Chat;

        Ok(Box::pin(try_stream! {
            // ACP agents execute tools internally. Goose never dispatches tool calls;
            // it only sees text, thoughts, and permission requests from the agent.
            //
            // In Chat mode (reject_all_tools), we suppress all text after a tool
            // starts because the agent may send tool results as AcpUpdate::Text,
            // bypassing the permission response.
            let mut suppress_text = false;

            while let Some(update) = rx.recv().await {
                match update {
                    AcpUpdate::Text(text) => {
                        if !suppress_text {
                            let message = Message::assistant().with_text(text);
                            yield (Some(message), None);
                        }
                    }
                    AcpUpdate::Thought(text) => {
                        let message = Message::assistant()
                            .with_thinking(text, "")
                            .with_visibility(true, false);
                        yield (Some(message), None);
                    }
                    AcpUpdate::ToolCallStart { id, .. } => {
                        if reject_all_tools {
                            suppress_text = true;
                            rejected_tool_calls.lock().await.insert(id);
                        }
                    }
                    AcpUpdate::ToolCallComplete { id, .. } => {
                        let is_error = rejected_tool_calls.lock().await.remove(&id);
                        if is_error {
                            let message = Message::assistant().with_text("Tool call was denied.");
                            yield (Some(message), None);
                        }
                    }
                    AcpUpdate::PermissionRequest { request, response_tx } => {
                        if let Some(decision) = permission_decision_from_mode(goose_mode) {
                            if decision.should_record_rejection() {
                                rejected_tool_calls.lock().await.insert(request.tool_call.tool_call_id.0.to_string());
                            }
                            let response = map_permission_response(&permission_mapping, &request, decision);
                            let _ = response_tx.send(response);
                            continue;
                        }

                        let request_id = request.tool_call.tool_call_id.0.to_string();
                        let (tx, rx) = oneshot::channel();

                        pending_confirmations
                            .lock()
                            .await
                            .insert(request_id.clone(), tx);

                        if let Some(action_required) = build_action_required_message(&request) {
                            yield (Some(action_required), None);
                        }

                        let confirmation = rx.await.unwrap_or(PermissionConfirmation {
                            principal_type: PrincipalType::Tool,
                            permission: Permission::Cancel,
                        });

                        pending_confirmations.lock().await.remove(&request_id);

                        let decision = PermissionDecision::from(confirmation.permission);
                        if decision.should_record_rejection() {
                            rejected_tool_calls.lock().await.insert(request.tool_call.tool_call_id.0.to_string());
                        }
                        let response = map_permission_response(&permission_mapping, &request, decision);
                        let _ = response_tx.send(response);
                    }
                    AcpUpdate::Complete(_reason) => {
                        break;
                    }
                    AcpUpdate::Error(e) => {
                        Err(ProviderError::RequestFailed(e))?;
                    }
                }
            }
        }))
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        let response = self.ensure_session(None).await?;
        Ok(response
            .models
            .map(|state| {
                state
                    .available_models
                    .iter()
                    .map(|m| m.model_id.0.to_string())
                    .collect()
            })
            .unwrap_or_default())
    }
}

impl Drop for AcpProvider {
    fn drop(&mut self) {
        // Join OS thread so session/close completes before runtime exits (reqwest InnerClientHandle pattern).
        self.tx.take();
        if let Some(h) = self.loop_thread.take() {
            if let Err(e) = h.join() {
                tracing::debug!("AcpClientLoop thread panicked: {e:?}");
            }
        }
    }
}

struct AcpClientLoop {
    config: AcpProviderConfig,
    goose_mode: Arc<Mutex<GooseMode>>,
    prompt_response_tx: Arc<Mutex<Option<mpsc::Sender<AcpUpdate>>>>,
}

impl AcpClientLoop {
    fn new(config: AcpProviderConfig, goose_mode: Arc<Mutex<GooseMode>>) -> Self {
        Self {
            config,
            goose_mode,
            prompt_response_tx: Arc::new(Mutex::new(None)),
        }
    }

    async fn spawn(
        self,
        mut rx: mpsc::Receiver<ClientRequest>,
        init_tx: oneshot::Sender<Result<InitializeResponse>>,
    ) {
        let child = match spawn_acp_process(&self.config).await {
            Ok(c) => c,
            Err(e) => {
                let _ = init_tx.send(Err(anyhow::anyhow!("{e}")));
                tracing::error!("failed to spawn ACP process: {e}");
                return;
            }
        };

        match self.run_with_child(child, &mut rx, init_tx).await {
            Ok(()) => tracing::debug!("ACP protocol loop exited cleanly"),
            Err(e) => tracing::error!(error = %e, "ACP protocol loop error"),
        }
    }

    async fn run_with_child(
        self,
        mut child: Child,
        rx: &mut mpsc::Receiver<ClientRequest>,
        init_tx: oneshot::Sender<Result<InitializeResponse>>,
    ) -> Result<()> {
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let transport = sacp::ByteStreams::new(stdin.compat_write(), stdout.compat());
        self.run(transport, rx, init_tx).await
    }

    async fn run<R, W>(
        self,
        transport: sacp::ByteStreams<W, R>,
        rx: &mut mpsc::Receiver<ClientRequest>,
        init_tx: oneshot::Sender<Result<InitializeResponse>>,
    ) -> Result<()>
    where
        R: futures::AsyncRead + Unpin + Send + 'static,
        W: futures::AsyncWrite + Unpin + Send + 'static,
    {
        let AcpClientLoop {
            config,
            goose_mode,
            prompt_response_tx,
        } = self;
        let notification_callback = config.notification_callback.clone();

        Client
            .builder()
            .on_receive_notification(
                {
                    let prompt_response_tx = prompt_response_tx.clone();
                    async move |notification: SessionNotification, _cx| {
                        if let Some(ref cb) = notification_callback {
                            cb(notification.clone());
                        }
                        // stream() reads goose_mode at call time, so it must
                        // reflect any prior set_mode before the next prompt.
                        match &notification.update {
                            SessionUpdate::CurrentModeUpdate(update) => {
                                if let Ok(mode) = GooseMode::from_str(&update.current_mode_id.0) {
                                    if let Ok(mut guard) = goose_mode.lock() {
                                        *guard = mode;
                                    }
                                }
                            }
                            SessionUpdate::ConfigOptionUpdate(update) => {
                                for opt in &update.config_options {
                                    if opt.category == Some(SessionConfigOptionCategory::Mode) {
                                        if let SessionConfigKind::Select(sel) = &opt.kind {
                                            if let Ok(mode) =
                                                GooseMode::from_str(&sel.current_value.0)
                                            {
                                                if let Ok(mut guard) = goose_mode.lock() {
                                                    *guard = mode;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        if let Some(tx) = prompt_response_tx
                            .lock()
                            .ok()
                            .as_ref()
                            .and_then(|g| g.as_ref())
                        {
                            match notification.update {
                                SessionUpdate::AgentMessageChunk(ContentChunk {
                                    content: ContentBlock::Text(TextContent { text, .. }),
                                    ..
                                }) => {
                                    let _ = tx.try_send(AcpUpdate::Text(text));
                                }
                                SessionUpdate::AgentThoughtChunk(ContentChunk {
                                    content: ContentBlock::Text(TextContent { text, .. }),
                                    ..
                                }) => {
                                    let _ = tx.try_send(AcpUpdate::Thought(text));
                                }
                                SessionUpdate::ToolCall(tool_call) => {
                                    let _ = tx.try_send(AcpUpdate::ToolCallStart {
                                        id: tool_call.tool_call_id.0.to_string(),
                                    });
                                }
                                SessionUpdate::ToolCallUpdate(update) => {
                                    if update.fields.status.is_some() {
                                        let _ = tx.try_send(AcpUpdate::ToolCallComplete {
                                            id: update.tool_call_id.0.to_string(),
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                        Ok(())
                    }
                },
                sacp::on_receive_notification!(),
            )
            .on_receive_request(
                {
                    let prompt_response_tx = prompt_response_tx.clone();
                    async move |request: RequestPermissionRequest, responder, _connection_cx| {
                        let (response_tx, response_rx) = oneshot::channel();

                        let handler = prompt_response_tx
                            .lock()
                            .ok()
                            .as_ref()
                            .and_then(|g| g.as_ref().cloned());
                        let tx = handler.ok_or_else(sacp::Error::internal_error)?;

                        if tx.is_closed() {
                            return Err(sacp::Error::internal_error());
                        }

                        tx.try_send(AcpUpdate::PermissionRequest {
                            request: Box::new(request),
                            response_tx,
                        })
                        .map_err(|_| sacp::Error::internal_error())?;

                        let response = response_rx.await.unwrap_or_else(|_| {
                            RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled)
                        });
                        responder.respond(response)
                    }
                },
                sacp::on_receive_request!(),
            )
            .connect_with(transport, async move |cx: ConnectionTo<Agent>| {
                handle_requests(config, cx, rx, prompt_response_tx, init_tx).await
            })
            .await?;

        Ok(())
    }
}

async fn spawn_acp_process(config: &AcpProviderConfig) -> Result<Child> {
    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);

    for key in &config.env_remove {
        cmd.env_remove(key);
    }

    for (key, value) in &config.env {
        cmd.env(key, value);
    }

    cmd.spawn().context("failed to spawn ACP process")
}

// sacp's connect_with unwraps the handler result, so we can't propagate
// send failures via ?. Log and continue when the receiver has dropped.
fn log_undelivered<E: std::fmt::Debug>(result: Result<(), E>, method: &str) {
    if let Err(e) = result {
        tracing::debug!(method, error = ?e, "response not delivered");
    }
}

async fn handle_requests(
    config: AcpProviderConfig,
    cx: ConnectionTo<Agent>,
    rx: &mut mpsc::Receiver<ClientRequest>,
    prompt_response_tx: Arc<Mutex<Option<mpsc::Sender<AcpUpdate>>>>,
    init_tx: oneshot::Sender<Result<InitializeResponse>>,
) -> Result<(), sacp::Error> {
    let mut init_tx = Some(init_tx);

    let init_response: InitializeResponse = cx
        .send_request(InitializeRequest::new(ProtocolVersion::LATEST))
        .block_task()
        .await
        .map_err(|err| {
            let message = format!("ACP {} failed: {err}", AGENT_METHOD_NAMES.initialize);
            // Attempt to send a specific error to the ctor waiting on init_rx;
            if let Some(tx) = init_tx.take() {
                let _ = tx.send(Err(anyhow::anyhow!(message.clone())));
            }
            sacp::Error::internal_error().data(message)
        })?;

    let supports_close = init_response
        .agent_capabilities
        .session_capabilities
        .close
        .is_some();
    let mcp_capabilities = init_response.agent_capabilities.mcp_capabilities.clone();
    if let Some(tx) = init_tx.take() {
        log_undelivered(tx.send(Ok(init_response)), AGENT_METHOD_NAMES.initialize);
    }

    let mut session_ids: Vec<SessionId> = Vec::new();

    while let Some(request) = rx.recv().await {
        match request {
            ClientRequest::NewSession { response_tx } => {
                if let Some(sid) =
                    handle_new_session_request(&config, &cx, &mcp_capabilities, response_tx).await
                {
                    session_ids.push(sid);
                }
            }
            ClientRequest::ListSessions { response_tx } => {
                let result: Result<ListSessionsResponse> = cx
                    .send_request(ListSessionsRequest::new())
                    .block_task()
                    .await
                    .map_err(anyhow::Error::from);
                log_undelivered(response_tx.send(result), AGENT_METHOD_NAMES.session_list);
            }
            ClientRequest::SetMode {
                session_id,
                mode_id,
                response_tx,
            } => {
                let result: Result<()> = cx
                    .send_request(SetSessionModeRequest::new(session_id, mode_id))
                    .block_task()
                    .await
                    .map(|_| ())
                    .map_err(anyhow::Error::from);
                log_undelivered(
                    response_tx.send(result),
                    AGENT_METHOD_NAMES.session_set_mode,
                );
            }
            ClientRequest::SetModel {
                session_id,
                model_id,
                response_tx,
            } => {
                let result: Result<()> = cx
                    .send_request(SetSessionModelRequest::new(session_id, model_id))
                    .block_task()
                    .await
                    .map(|_| ())
                    .map_err(anyhow::Error::from);
                log_undelivered(
                    response_tx.send(result),
                    AGENT_METHOD_NAMES.session_set_model,
                );
            }
            ClientRequest::SetConfigOption {
                session_id,
                config_id,
                value,
                response_tx,
            } => {
                let value_id = sacp::schema::SessionConfigValueId::new(value);
                let req = SetSessionConfigOptionRequest::new(session_id, config_id, value_id);
                let result: Result<()> = cx
                    .send_request(req)
                    .block_task()
                    .await
                    .map(|_| ())
                    .map_err(anyhow::Error::from);
                log_undelivered(
                    response_tx.send(result),
                    AGENT_METHOD_NAMES.session_set_config_option,
                );
            }
            ClientRequest::Untyped {
                method,
                params,
                response_tx,
            } => {
                let result: Result<serde_json::Value> =
                    match sacp::UntypedMessage::new(&method, params) {
                        Ok(msg) => cx
                            .send_request(msg)
                            .block_task()
                            .await
                            .map_err(anyhow::Error::from),
                        Err(e) => Err(anyhow::Error::from(e)),
                    };
                log_undelivered(response_tx.send(result), &method);
            }
            ClientRequest::Prompt {
                session_id,
                content,
                response_tx,
            } => {
                *prompt_response_tx.lock().unwrap() = Some(response_tx.clone());

                let response: Result<PromptResponse, _> = cx
                    .send_request(PromptRequest::new(session_id, content))
                    .block_task()
                    .await;

                match response {
                    Ok(r) => {
                        log_undelivered(
                            response_tx.try_send(AcpUpdate::Complete(r.stop_reason)),
                            AGENT_METHOD_NAMES.session_prompt,
                        );
                    }
                    Err(e) => {
                        log_undelivered(
                            response_tx.try_send(AcpUpdate::Error(e.to_string())),
                            AGENT_METHOD_NAMES.session_prompt,
                        );
                    }
                }

                *prompt_response_tx.lock().unwrap() = None;
            }
        }
    }

    // After loop exits (channel closed by Drop):
    if supports_close {
        for session_id in session_ids {
            if let Err(e) = cx
                .send_request(CloseSessionRequest::new(session_id.clone()))
                .block_task()
                .await
            {
                tracing::debug!(method = AGENT_METHOD_NAMES.session_close, session_id = %session_id, error = %e, "failed on shutdown");
            }
        }
    }

    Ok(())
}

async fn handle_new_session_request(
    config: &AcpProviderConfig,
    cx: &ConnectionTo<Agent>,
    mcp_capabilities: &McpCapabilities,
    response_tx: oneshot::Sender<Result<NewSessionResponse>>,
) -> Option<SessionId> {
    let mcp_servers = filter_supported_servers(&config.mcp_servers, mcp_capabilities);
    let session: Result<NewSessionResponse, _> = cx
        .send_request(NewSessionRequest::new(config.work_dir.clone()).mcp_servers(mcp_servers))
        .block_task()
        .await;

    let result = match session {
        Ok(session) => apply_session_mode(config, cx, session).await,
        Err(err) => Err(anyhow::anyhow!(
            "ACP {} failed: {err}",
            AGENT_METHOD_NAMES.session_new
        )),
    };

    let session_id = match &result {
        Ok(s) => Some(s.session_id.clone()),
        Err(e) => {
            tracing::debug!(method = AGENT_METHOD_NAMES.session_new, error = %e, "failed");
            None
        }
    };
    if let Err(e) = response_tx.send(result) {
        tracing::debug!(method = AGENT_METHOD_NAMES.session_new, error = ?e, "response not delivered");
    }
    session_id
}

async fn apply_session_mode(
    config: &AcpProviderConfig,
    cx: &ConnectionTo<Agent>,
    session: NewSessionResponse,
) -> Result<NewSessionResponse> {
    if let (Some(mode_id), Some(modes)) = (config.session_mode_id.clone(), session.modes.as_ref()) {
        if modes.current_mode_id.0.as_ref() != mode_id.as_str() {
            let available: Vec<String> = modes
                .available_modes
                .iter()
                .map(|mode| mode.id.0.to_string())
                .collect();

            if !available.iter().any(|id| id == &mode_id) {
                return Err(anyhow::anyhow!(
                    "Requested mode '{}' not offered by agent. Available modes: {}",
                    mode_id,
                    available.join(", ")
                ));
            }
            let _: SetSessionModeResponse = cx
                .send_request(SetSessionModeRequest::new(
                    session.session_id.clone(),
                    mode_id,
                ))
                .block_task()
                .await
                .map_err(|err| {
                    anyhow::anyhow!(
                        "ACP agent rejected {}: {err}",
                        AGENT_METHOD_NAMES.session_set_mode
                    )
                })?;
        }
    }

    Ok(session)
}

pub fn extension_configs_to_mcp_servers(configs: &[ExtensionConfig]) -> Vec<McpServer> {
    let mut servers = Vec::new();

    for config in configs {
        match config {
            ExtensionConfig::StreamableHttp {
                name, uri, headers, ..
            } => {
                let http_headers = headers
                    .iter()
                    .map(|(key, value)| HttpHeader::new(key, value))
                    .collect();
                servers.push(McpServer::Http(
                    McpServerHttp::new(name, uri).headers(http_headers),
                ));
            }
            ExtensionConfig::Stdio {
                name,
                cmd,
                args,
                envs,
                ..
            } => {
                let env_vars = envs
                    .get_env()
                    .into_iter()
                    .map(|(key, value)| EnvVariable::new(key, value))
                    .collect();

                servers.push(McpServer::Stdio(
                    McpServerStdio::new(name, cmd)
                        .args(args.clone())
                        .env(env_vars),
                ));
            }
            ExtensionConfig::Sse { name, .. } => {
                tracing::debug!(name, "skipping SSE extension, migrate to streamable_http");
            }
            _ => {}
        }
    }

    servers
}

fn filter_supported_servers(
    servers: &[McpServer],
    capabilities: &McpCapabilities,
) -> Vec<McpServer> {
    servers
        .iter()
        .filter(|server| match server {
            McpServer::Http(http) => {
                if !capabilities.http {
                    tracing::debug!(
                        name = http.name,
                        "skipping HTTP server, agent lacks capability"
                    );
                    false
                } else {
                    true
                }
            }
            McpServer::Sse(sse) => {
                tracing::debug!(name = sse.name, "skipping SSE server, unsupported");
                false
            }
            _ => true,
        })
        .cloned()
        .collect()
}

fn messages_to_prompt(messages: &[Message]) -> Vec<ContentBlock> {
    let mut content_blocks = Vec::new();

    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User && m.is_agent_visible());

    if let Some(message) = last_user {
        for content in &message.content {
            match content {
                MessageContent::Text(text) => {
                    content_blocks.push(ContentBlock::Text(TextContent::new(text.text.clone())));
                }
                MessageContent::Image(image) => {
                    content_blocks.push(ContentBlock::Image(ImageContent::new(
                        &image.data,
                        &image.mime_type,
                    )));
                }
                _ => {}
            }
        }
    }

    content_blocks
}

fn build_action_required_message(request: &RequestPermissionRequest) -> Option<Message> {
    let tool_title = request
        .tool_call
        .fields
        .title
        .clone()
        .unwrap_or_else(|| "Tool".to_string());

    let arguments = request
        .tool_call
        .fields
        .raw_input
        .as_ref()
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();

    let prompt = request
        .tool_call
        .fields
        .content
        .as_ref()
        .and_then(|content| {
            content.iter().find_map(|c| match c {
                ToolCallContent::Content(val) => match &val.content {
                    ContentBlock::Text(text) => Some(text.text.clone()),
                    _ => None,
                },
                _ => None,
            })
        });

    Some(
        Message::assistant()
            .with_action_required(
                request.tool_call.tool_call_id.0.to_string(),
                tool_title,
                arguments,
                prompt,
            )
            .user_only(),
    )
}

fn permission_decision_from_mode(goose_mode: GooseMode) -> Option<PermissionDecision> {
    match goose_mode {
        GooseMode::Auto => Some(PermissionDecision::AllowOnce),
        GooseMode::Chat => Some(PermissionDecision::RejectOnce),
        GooseMode::Approve | GooseMode::SmartApprove => None,
    }
}

// TODO: ID mapping is in-memory only — sessions from prior runs or other clients are dropped.
// Persisting requires a schema change to map goose↔ACP IDs in the session DB.
fn map_sessions_to_goose_ids(
    response: ListSessionsResponse,
    acp_to_goose: &HashMap<String, String>,
) -> ListSessionsResponse {
    let sessions = response
        .sessions
        .into_iter()
        .filter_map(|mut info| {
            let goose_id = acp_to_goose.get(info.session_id.0.as_ref())?;
            info.session_id = SessionId::new(goose_id.clone());
            Some(info)
        })
        .collect();
    ListSessionsResponse::new(sessions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::extension::Envs;
    use sacp::schema::SessionInfo;
    use test_case::test_case;

    #[test_case(
        ExtensionConfig::Stdio {
            name: "github".into(),
            description: String::new(),
            cmd: "/path/to/github-mcp-server".into(),
            args: vec!["stdio".into()],
            envs: Envs::new([("GITHUB_PERSONAL_ACCESS_TOKEN".into(), "ghp_xxxxxxxxxxxx".into())].into()),
            env_keys: vec![],
            timeout: None,
            bundled: Some(false),
            available_tools: vec![],
        },
        vec![
            McpServer::Stdio(
                McpServerStdio::new("github", "/path/to/github-mcp-server")
                    .args(vec!["stdio".into()])
                    .env(vec![EnvVariable::new("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_xxxxxxxxxxxx")])
            )
        ]
        ; "stdio_converts_to_mcpserver_stdio"
    )]
    #[test_case(
        ExtensionConfig::StreamableHttp {
            name: "github".into(),
            description: String::new(),
            uri: "https://api.githubcopilot.com/mcp/".into(),
            envs: Envs::default(),
            env_keys: vec![],
            headers: HashMap::from([("Authorization".into(), "Bearer ghp_xxxxxxxxxxxx".into())]),
            timeout: None,
            bundled: Some(false),
            available_tools: vec![],
        },
        vec![
            McpServer::Http(
                McpServerHttp::new("github", "https://api.githubcopilot.com/mcp/")
                    .headers(vec![HttpHeader::new("Authorization", "Bearer ghp_xxxxxxxxxxxx")])
            )
        ]
        ; "streamable_http_converts_to_mcpserver_http_when_capable"
    )]
    fn test_extension_configs_to_mcp_servers(config: ExtensionConfig, expected: Vec<McpServer>) {
        let result = extension_configs_to_mcp_servers(&[config]);
        assert_eq!(result.len(), expected.len(), "server count mismatch");
        for (a, e) in result.iter().zip(expected.iter()) {
            match (a, e) {
                (McpServer::Stdio(actual), McpServer::Stdio(expected)) => {
                    assert_eq!(actual.name, expected.name);
                    assert_eq!(actual.command, expected.command);
                    assert_eq!(actual.args, expected.args);
                    assert_eq!(actual.env.len(), expected.env.len());
                }
                (McpServer::Http(actual), McpServer::Http(expected)) => {
                    assert_eq!(actual.name, expected.name);
                    assert_eq!(actual.url, expected.url);
                    assert_eq!(actual.headers.len(), expected.headers.len());
                }
                _ => panic!("server type mismatch"),
            }
        }
    }

    #[test]
    fn test_sse_skips() {
        let config = ExtensionConfig::Sse {
            name: "test-sse".into(),
            description: String::new(),
            uri: Some("https://example.com/sse".into()),
        };
        let result = extension_configs_to_mcp_servers(&[config]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_supported_servers_skips_http_without_capability() {
        let config = ExtensionConfig::StreamableHttp {
            name: "github".into(),
            description: String::new(),
            uri: "https://api.githubcopilot.com/mcp/".into(),
            envs: Envs::default(),
            env_keys: vec![],
            headers: HashMap::from([("Authorization".into(), "Bearer ghp_xxxxxxxxxxxx".into())]),
            timeout: None,
            bundled: Some(false),
            available_tools: vec![],
        };

        let servers = extension_configs_to_mcp_servers(&[config]);
        let filtered = filter_supported_servers(&servers, &McpCapabilities::default());
        assert!(filtered.is_empty());
    }

    #[test_case(
        ListSessionsResponse::new(vec![
            SessionInfo::new(SessionId::new("20260318_1"), "/Users/codefromthecrypt/oss/goose-2")
                .title("Fix login bug".to_string())
                .updated_at("2026-03-18T07:02:42.549655Z".to_string()),
            SessionInfo::new(SessionId::new("20260318_2"), "/tmp/test-acpx")
                .title("Add caching layer".to_string())
                .updated_at("2026-03-18T07:05:01.123Z".to_string()),
        ]),
        HashMap::from([
            ("20260318_1".to_string(), "goose-session-1".to_string()),
            ("20260318_2".to_string(), "goose-session-2".to_string()),
        ]),
        ListSessionsResponse::new(vec![
            SessionInfo::new(SessionId::new("goose-session-1"), "/Users/codefromthecrypt/oss/goose-2")
                .title("Fix login bug".to_string())
                .updated_at("2026-03-18T07:02:42.549655Z".to_string()),
            SessionInfo::new(SessionId::new("goose-session-2"), "/tmp/test-acpx")
                .title("Add caching layer".to_string())
                .updated_at("2026-03-18T07:05:01.123Z".to_string()),
        ])
        ; "all sessions mapped with all fields preserved"
    )]
    #[test_case(
        ListSessionsResponse::new(vec![
            SessionInfo::new(SessionId::new("20260318_1"), "/Users/codefromthecrypt/oss/goose-2")
                .title("Fix login bug".to_string()),
            SessionInfo::new(SessionId::new("other-agent-session"), "/tmp/other")
                .title("Not our session".to_string()),
        ]),
        HashMap::from([
            ("20260318_1".to_string(), "goose-session-1".to_string()),
        ]),
        ListSessionsResponse::new(vec![
            SessionInfo::new(SessionId::new("goose-session-1"), "/Users/codefromthecrypt/oss/goose-2")
                .title("Fix login bug".to_string()),
        ])
        ; "unmapped sessions filtered out"
    )]
    #[test_case(
        ListSessionsResponse::new(vec![
            SessionInfo::new(SessionId::new("20260318_1"), "/Users/codefromthecrypt/oss/goose-2")
                .title("ACP Session".to_string())
                .updated_at("2026-03-18T01:29:02.141700Z".to_string()),
        ]),
        HashMap::new(),
        ListSessionsResponse::new(vec![])
        ; "empty map returns empty list"
    )]
    fn test_map_sessions_to_goose_ids(
        response: ListSessionsResponse,
        acp_to_goose: HashMap<String, String>,
        expected: ListSessionsResponse,
    ) {
        let result = map_sessions_to_goose_ids(response, &acp_to_goose);
        assert_eq!(result, expected);
    }

    #[test_case(GooseMode::Auto => Some(PermissionDecision::AllowOnce) ; "auto allows")]
    #[test_case(GooseMode::Chat => Some(PermissionDecision::RejectOnce) ; "chat rejects")]
    #[test_case(GooseMode::Approve => None ; "approve defers")]
    #[test_case(GooseMode::SmartApprove => None ; "smart_approve defers")]
    fn test_permission_decision_from_mode(mode: GooseMode) -> Option<PermissionDecision> {
        permission_decision_from_mode(mode)
    }
}
