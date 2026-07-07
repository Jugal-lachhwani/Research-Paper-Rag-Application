const API_BASE_URL = 'http://localhost:8002'; // Make sure this matches your FastAPI server port

// DOM Elements
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatForm = document.getElementById('chat-form');
const messagesWrapper = document.getElementById('messages-wrapper');
const welcomeScreen = document.getElementById('welcome-screen');
const chatHistoryList = document.getElementById('chat-history-list');
const newChatBtn = document.getElementById('new-chat-btn');
const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const sidebar = document.querySelector('.sidebar');
const chatContainer = document.getElementById('chat-container');

// State
let currentChatId = null;

// Configure marked.js to use target="_blank" for links
const renderer = new marked.Renderer();
renderer.link = function(href, title, text) {
    return `<a href="${href}" target="_blank" rel="noopener noreferrer" title="${title || ''}">${text}</a>`;
};
marked.setOptions({ renderer });

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendBtn.disabled = this.value.trim().length === 0;
});

// Submit on Enter (Shift+Enter for new line)
chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (this.value.trim().length > 0) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
});

// Toggle mobile sidebar
toggleSidebarBtn?.addEventListener('click', () => {
    sidebar.classList.toggle('open');
});

// Start new chat
newChatBtn.addEventListener('click', () => {
    currentChatId = null;
    messagesWrapper.innerHTML = '';
    messagesWrapper.style.display = 'none';
    welcomeScreen.style.display = 'flex';
    document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('active'));
    chatInput.focus();
    if (window.innerWidth <= 768) sidebar.classList.remove('open');
});

// Fetch chat history list from backend
async function loadChatHistoryList() {
    try {
        const response = await fetch(`${API_BASE_URL}/chats`);
        if (!response.ok) throw new Error('Failed to fetch chats');
        
        const chatIds = await response.json();
        chatHistoryList.innerHTML = '';
        
        chatIds.forEach(id => {
            const li = document.createElement('li');
            li.className = 'chat-item';
            li.innerHTML = `<i class="ph ph-chat-circle"></i> Chat ${id.substring(0, 8)}...`;
            li.onclick = () => loadChat(id, li);
            if (id === currentChatId) li.classList.add('active');
            chatHistoryList.appendChild(li);
        });
    } catch (error) {
        console.error('Error loading chat history list:', error);
    }
}

// Load a specific chat
async function loadChat(chatId, listItemElement) {
    currentChatId = chatId;
    
    // Update active state in sidebar
    document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('active'));
    if (listItemElement) listItemElement.classList.add('active');
    
    if (window.innerWidth <= 768) sidebar.classList.remove('open');

    // Show loading UI
    welcomeScreen.style.display = 'none';
    messagesWrapper.style.display = 'flex';
    messagesWrapper.innerHTML = `
        <div class="message">
            <div class="loading-indicator">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                Loading chat...
            </div>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/chats/${chatId}/state`);
        if (!response.ok) throw new Error('Failed to load chat');
        
        const data = await response.json();
        const messages = data.state.messages || [];
        
        messagesWrapper.innerHTML = '';
        messages.forEach(msg => {
            if (msg.role === 'user') {
                appendMessage(msg.content, 'user');
            } else if (msg.role === 'assistant') {
                // In a loaded state, we might not have the exact references mapped per message easily,
                // but we display the text content.
                appendMessage(msg.content, 'bot');
            }
        });
        scrollToBottom();
    } catch (error) {
        console.error('Error loading chat state:', error);
        messagesWrapper.innerHTML = `<div class="message"><div style="color: #ff5e5e;">Error loading chat history.</div></div>`;
    }
}

// Handle Form Submit
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // UI Updates
    welcomeScreen.style.display = 'none';
    messagesWrapper.style.display = 'flex';
    appendMessage(message, 'user');
    scrollToBottom();

    // Add loading indicator
    const loadingId = appendLoading();
    scrollToBottom();

    try {
        const requestBody = { message: message };
        if (currentChatId) {
            requestBody.chat_id = currentChatId;
        }

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Server error');

        const data = await response.json();
        
        // Remove loading
        document.getElementById(loadingId).remove();
        
        // Update current chat ID if it was newly created
        if (!currentChatId && data.chat_id) {
            currentChatId = data.chat_id;
            loadChatHistoryList(); // Refresh sidebar to show the new chat
        }

        // Append bot response with references
        appendMessage(data.answer, 'bot', data.references);
        scrollToBottom();

    } catch (error) {
        console.error('Error sending message:', error);
        document.getElementById(loadingId).remove();
        appendMessage('Sorry, an error occurred while connecting to the server.', 'bot');
        scrollToBottom();
    }
});

function appendMessage(content, sender, references = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    let htmlContent = '';
    if (sender === 'bot') {
        htmlContent = marked.parse(content);
        
        // Add references if they exist
        if (references && references.length > 0) {
            htmlContent += `<div class="references-container">
                <div class="references-title">References</div>
                <div class="references-list">
                    ${references.map((ref, idx) => {
                        const validUrl = getValidUrl(ref);
                        const displayName = formatReferenceName(ref);
                        return `<a href="${validUrl}" target="_blank" class="reference-tag">[${idx + 1}] ${displayName}</a>`;
                    }).join('')}
                </div>
            </div>`;
        }
    } else {
        // Simple escape for user text
        htmlContent = `<p>${content.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</p>`;
    }

    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="avatar">${sender === 'user' ? '<i class="ph ph-user"></i>' : '<i class="ph-fill ph-atom"></i>'}</div>
            <div class="text-content">${htmlContent}</div>
        </div>
    `;
    
    messagesWrapper.appendChild(messageDiv);
}

function appendLoading() {
    const id = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = id;
    loadingDiv.className = 'message bot';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="avatar"><i class="ph-fill ph-atom"></i></div>
            <div class="text-content" style="display:flex; align-items:center;">
                <div class="loading-indicator">
                    <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                </div>
            </div>
        </div>
    `;
    messagesWrapper.appendChild(loadingDiv);
    return id;
}

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// Utility to get a valid URL for the href
function getValidUrl(ref) {
    if (ref.startsWith('http://') || ref.startsWith('https://')) {
        return ref;
    }
    // Assume it's an ArXiv ID if there's no http scheme
    return `https://arxiv.org/abs/${ref}`;
}

// Utility to make long URLs or IDs look cleaner in tags
function formatReferenceName(ref) {
    try {
        if (!ref.startsWith('http')) {
            return `arXiv:${ref}`;
        }
        const urlObj = new URL(ref);
        let hostname = urlObj.hostname.replace('www.', '');
        if (hostname.length > 20) hostname = hostname.substring(0, 17) + '...';
        return hostname;
    } catch(e) {
        return ref;
    }
}

// Init
loadChatHistoryList();
