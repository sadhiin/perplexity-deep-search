// DOM elements
const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const chatList = document.getElementById('chat-list');
const newChatBtn = document.getElementById('new-chat-btn');
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.querySelector('.sidebar');
const charCount = document.querySelector('.char-count');
const themeToggle = document.getElementById('theme-toggle');

const MESSAGE_TYPES = {
    user: { label: 'You' },
    assistant: { label: 'Assistant' },
    system: { label: 'System' },
    research: { label: 'Research update' },
};

function normalizeMessageType(type) {
    if (type === 'bot') {
        return 'assistant';
    }
    return MESSAGE_TYPES[type] ? type : 'assistant';
}

function createMessageRecord(content, type, timestamp) {
    return {
        content,
        type: normalizeMessageType(type),
        timestamp: timestamp || new Date().toISOString(),
    };
}

function truncateText(text, maxLength = 80) {
    if (!text) return '';
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

// Markdown/HTML helpers
function escapeHtml(text = '') {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function autoLinkUrls(html) {
    return html.replace(/https?:\/\/[^\s<]+/g, (url, offset, full) => {
        const lastOpen = full.lastIndexOf('<', offset);
        const lastClose = full.lastIndexOf('>', offset);
        if (lastOpen > lastClose) {
            return url;
        }
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
    });
}

function applyInlineFormatting(safeText) {
    let formatted = safeText;

    formatted = formatted.replace(/!\[([^\]]*?)\]\((https?:\/\/[^\s)]+)\)/g, (_, alt, url) => {
        const altText = alt || 'Image';
        return `<span class="message-image"><img src="${url}" alt="${altText}" loading="lazy" /></span>`;
    });

    formatted = formatted.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) =>
        `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
    );

    formatted = formatted.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    formatted = autoLinkUrls(formatted);

    return formatted.replace(/\n/g, '<br>');
}

function formatMessageContent(text = '') {
    const codeBlockRegex = /```([\s\S]*?)```/g;
    let lastIndex = 0;
    let result = '';
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
        const segment = text.slice(lastIndex, match.index);
        if (segment) {
            result += applyInlineFormatting(escapeHtml(segment));
        }

        const codeContent = escapeHtml(match[1].trim());
        result += `<pre><code>${codeContent}</code></pre>`;
        lastIndex = match.index + match[0].length;
    }

    const remaining = text.slice(lastIndex);
    if (remaining) {
        result += applyInlineFormatting(escapeHtml(remaining));
    }

    return result || '';
}

// State management
let currentChatId = null;
let chats = JSON.parse(localStorage.getItem('chats')) || [];
let isTyping = false;
let currentTheme = localStorage.getItem('theme') || 'light';

// Configuration
const API_BASE_URL = window.API_BASE_URL || 'http://localhost:8000'; // Will be set by nginx or fallback to localhost

// Initialize the app
function init() {
    loadChats();
    setupEventListeners();
    if (chats.length === 0) {
        createNewChat();
    } else {
        currentChatId = chats[0].id;
        loadChat(currentChatId);
    }
    adjustTextareaHeight();
    applyTheme();

    console.log('AI Chat Assistant initialized');
}

// Event listeners
function setupEventListeners() {
    messageForm.addEventListener('submit', handleMessageSubmit);
    messageInput.addEventListener('input', handleInputChange);
    messageInput.addEventListener('keydown', handleKeyDown);
    newChatBtn.addEventListener('click', createNewChat);
    menuToggle.addEventListener('click', toggleSidebar);
    themeToggle.addEventListener('click', toggleTheme);

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && !sidebar.contains(e.target) && e.target !== menuToggle) {
            sidebar.classList.remove('open');
        }
    });
}

// Handle message form submission
async function handleMessageSubmit(e) {
    e.preventDefault();
    const message = messageInput.value.trim();

    if (!message || isTyping) return;

    // Add user message
    const userRecord = createMessageRecord(message, 'user');
    addMessage(userRecord.content, userRecord.type, userRecord.timestamp);
    updateCurrentChat(userRecord);
    messageInput.value = '';
    updateCharCount();
    adjustTextareaHeight();

    // Show typing indicator
    showTypingIndicator();

    // Provide research progress update
    const researchRecord = createMessageRecord(
        `Researching “${truncateText(message, 120)}”…`,
        'research'
    );
    addMessage(researchRecord.content, researchRecord.type, researchRecord.timestamp);
    updateCurrentChat(researchRecord);

    try {
        // Simulate AI response (replace with actual API call)
        const response = await getAIResponse(message);
        hideTypingIndicator();
        const assistantRecord = createMessageRecord(response, 'assistant');
        addMessage(assistantRecord.content, assistantRecord.type, assistantRecord.timestamp);
        updateCurrentChat(assistantRecord);
    } catch (error) {
        hideTypingIndicator();
        const errorRecord = createMessageRecord(
            'Sorry, I encountered an error. Please try again.',
            'system'
        );
        addMessage(errorRecord.content, errorRecord.type, errorRecord.timestamp);
        updateCurrentChat(errorRecord);
        console.error('Error getting AI response:', error);
    }

    // Update chat in history
    saveChats();
}

// Handle input changes
function handleInputChange() {
    updateCharCount();
    adjustTextareaHeight();
    updateSendButtonState();
}

// Handle keyboard events
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        messageForm.dispatchEvent(new Event('submit'));
    }
}

// Update character count
function updateCharCount() {
    const count = messageInput.value.length;
    charCount.textContent = `${count}/2000`;
}

// Adjust textarea height
function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// Update send button state
function updateSendButtonState() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isTyping;
}

// Add message to chat
function addMessage(content, type, timestamp) {
    const normalizedType = normalizeMessageType(type);
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${normalizedType}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    const roleMeta = MESSAGE_TYPES[normalizedType];
    if (roleMeta?.label) {
        const label = document.createElement('div');
        label.className = 'message-role';
        label.textContent = roleMeta.label;
        messageContent.appendChild(label);
    }

    const messageBody = document.createElement('div');
    messageBody.className = 'message-text';
    messageBody.innerHTML = formatMessageContent(content);
    messageContent.appendChild(messageBody);

    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    const timeValue = timestamp ? new Date(timestamp) : new Date();
    messageTime.textContent = timeValue.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    isTyping = true;
    updateSendButtonState();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typing-indicator';

    typingDiv.innerHTML = `
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        <span>AI is thinking...</span>
    `;

    messagesContainer.appendChild(typingDiv);
    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    isTyping = false;
    updateSendButtonState();

    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Simulate AI response (replace with actual API call)
async function getAIResponse(message) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    // Mock responses based on input
    const responses = [
        `I understand you're asking about "${message}". Let me help you with that. Based on my knowledge, here's what I can tell you...`,
        `That's an interesting question about "${message}". From what I know, the key points are...`,
        `Regarding "${message}", I'd be happy to provide some insights. Here's what you should know...`,
        `Great question! When it comes to "${message}", there are several important aspects to consider...`
    ];

    return responses[Math.floor(Math.random() * responses.length)];
}

// Create new chat
function createNewChat() {
    const chatId = Date.now().toString();
    const welcomeRecord = createMessageRecord(
        "Welcome to AI Chat. I'm ready to help you research, analyze, and discuss any topic.",
        'system'
    );
    const newChat = {
        id: chatId,
        title: 'New Chat',
        messages: [welcomeRecord],
        createdAt: welcomeRecord.timestamp,
        lastMessageAt: welcomeRecord.timestamp
    };

    chats.unshift(newChat);
    currentChatId = chatId;

    updateChatList();
    saveChats();
    loadChat(chatId);
}

// Update current chat with new message
function updateCurrentChat(messageRecord) {
    if (!messageRecord) return;
    if (typeof messageRecord === 'string') {
        messageRecord = createMessageRecord(messageRecord, 'user');
    }

    const chat = chats.find(c => c.id === currentChatId);
    if (!chat) return;

    chat.messages.push(messageRecord);
    chat.lastMessageAt = messageRecord.timestamp || new Date().toISOString();

    if (chat.title === 'New Chat' && messageRecord.type === 'user') {
        const preview = truncateText(messageRecord.content, 30);
        chat.title = preview || chat.title;
    }

    updateChatList();
}

// Update chat list in sidebar
function updateChatList() {
    chatList.innerHTML = '';

    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = `chat-item ${chat.id === currentChatId ? 'active' : ''}`;
        chatItem.onclick = () => loadChat(chat.id);

        chatItem.innerHTML = `
            <div class="chat-item-main">
                <div class="chat-item-title">${chat.title}</div>
                <div class="chat-item-time">${formatTime(chat.lastMessageAt)}</div>
            </div>
            <div class="chat-item-actions">
                <button class="chat-item-btn rename" title="Rename chat" aria-label="Rename chat">
                    ✏️
                </button>
                <button class="chat-item-btn delete" title="Delete chat" aria-label="Delete chat">
                    🗑️
                </button>
            </div>
        `;

        const renameBtn = chatItem.querySelector('.chat-item-btn.rename');
        renameBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            renameChat(chat.id);
        });

        const deleteBtn = chatItem.querySelector('.chat-item-btn.delete');
        deleteBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            deleteChat(chat.id);
        });

        chatList.appendChild(chatItem);
    });
}

// Load a specific chat
function loadChat(chatId) {
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;

    currentChatId = chatId;

    // Clear messages
    messagesContainer.innerHTML = '';

    if (chat.messages.length === 0) {
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-content">
                    <h2>Start chatting</h2>
                    <p>Ask me anything! I'm here to help you with information, analysis, and conversation.</p>
                </div>
            </div>
        `;
    } else {
        chat.messages.forEach(msg => {
            addMessage(msg.content, msg.type, msg.timestamp);
        });
    }

    updateChatList();
}

// Format time for chat list
function formatTime(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;

    return date.toLocaleDateString();
}

// Toggle sidebar on mobile
function toggleSidebar() {
    sidebar.classList.toggle('open');
}

// Theme management functions
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme();
    saveTheme();
}

function applyTheme() {
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = themeToggle.querySelector('svg path');
    if (currentTheme === 'dark') {
        // Sun icon for light mode
        icon.setAttribute('d', 'M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM18.894 6.166a.75.75 0 00-1.06-1.06l-1.591 1.59a.75.75 0 101.06 1.061l1.591-1.59zM21.75 12a.75.75 0 01-.75.75h-2.25a.75.75 0 010-1.5H21a.75.75 0 01.75.75zM17.834 18.894a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 10-1.061 1.06l1.59 1.591zM12 18a.75.75 0 01.75.75V21a.75.75 0 01-1.5 0v-2.25A.75.75 0 0112 18zM7.758 17.303a.75.75 0 00-1.061-1.06l-1.591 1.59a.75.75 0 001.06 1.061l1.591-1.59zM6 12a.75.75 0 01-.75.75H3a.75.75 0 010-1.5h2.25A.75.75 0 016 12zM6.697 7.757a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 00-1.061 1.06l1.59 1.591z');
    } else {
        // Moon icon for dark mode
        icon.setAttribute('d', 'M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z');
    }
}

function saveTheme() {
    localStorage.setItem('theme', currentTheme);
}

// Scroll to bottom of messages
function scrollToBottom() {
    setTimeout(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 100);
}

// Load chats from localStorage
function loadChats() {
    chats = JSON.parse(localStorage.getItem('chats')) || [];

    chats.forEach(chat => {
        chat.messages = (chat.messages || []).map(msg => ({
            content: msg.content || '',
            type: normalizeMessageType(msg.type),
            timestamp: msg.timestamp || new Date().toISOString(),
        }));
        if (!chat.lastMessageAt && chat.messages.length) {
            chat.lastMessageAt = chat.messages[chat.messages.length - 1].timestamp;
        }
    });

    updateChatList();
}

// Save chats to localStorage
function saveChats() {
    localStorage.setItem('chats', JSON.stringify(chats));
}

// Rename a chat conversation
function renameChat(chatId) {
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;

    const newTitle = prompt('Enter a new name for this chat:', chat.title || 'New Chat');
    if (newTitle === null) return;

    const trimmed = newTitle.trim();
    if (!trimmed || trimmed === chat.title) return;

    chat.title = trimmed;
    saveChats();
    updateChatList();
}

// Delete a chat and fallback to another conversation if needed
function deleteChat(chatId) {
    const index = chats.findIndex(c => c.id === chatId);
    if (index === -1) return;

    const confirmed = window.confirm('Delete this chat? This cannot be undone.');
    if (!confirmed) return;

    const wasActive = chats[index].id === currentChatId;
    chats.splice(index, 1);
    saveChats();

    if (chats.length === 0) {
        currentChatId = null;
        messagesContainer.innerHTML = '';
        createNewChat();
        return;
    }

    if (wasActive) {
        loadChat(chats[0].id);
    } else {
        updateChatList();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Export functions for potential testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        addMessage,
        getAIResponse,
        createNewChat,
        updateCurrentChat,
        renameChat,
        deleteChat
    };
}
