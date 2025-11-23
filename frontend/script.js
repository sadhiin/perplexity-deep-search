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
const bookmarkList = document.getElementById('bookmark-list');
const bookmarkCount = document.getElementById('bookmark-count');
const chatSearchInput = document.getElementById('chat-search');

const MESSAGE_TYPES = {
    user: { label: 'You' },
    assistant: { label: 'Assistant' },
    system: { label: 'System' },
    research: { label: 'Research update' },
};

let chatSearchTerm = '';

function generateMessageId() {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function normalizeMessageType(type) {
    if (type === 'bot') {
        return 'assistant';
    }
    return MESSAGE_TYPES[type] ? type : 'assistant';
}

function createMessageRecord(content, type, timestamp) {
    return {
        id: generateMessageId(),
        content,
        type: normalizeMessageType(type),
        timestamp: timestamp || new Date().toISOString(),
        isBookmarked: false,
        reaction: null,
    };
}

function truncateText(text, maxLength = 80) {
    if (!text) return '';
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlightSearchMatch(text) {
    const safe = escapeHtml(text || '');
    const term = chatSearchTerm.trim();
    if (!term) return safe;
    const regex = new RegExp(escapeRegExp(term), 'ig');
    return safe.replace(regex, (match) => `<mark>${match}</mark>`);
}

function chatMatchesSearch(chat, lowerTerm) {
    if (!lowerTerm) return true;
    const title = (chat.title || '').toLowerCase();
    if (title.includes(lowerTerm)) return true;
    return (chat.messages || []).some((msg) =>
        (msg.content || '').toLowerCase().includes(lowerTerm)
    );
}

function getChatSnippet(chat) {
    if (!chat.messages || !chat.messages.length) {
        return '';
    }
    const last = chat.messages[chat.messages.length - 1];
    return truncateText(last.content || '', 60);
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
    if (chatSearchInput) {
        chatSearchInput.addEventListener('input', handleChatSearch);
    }

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
    addMessage(userRecord);
    updateCurrentChat(userRecord);
    messageInput.value = '';
    updateCharCount();
    adjustTextareaHeight();

    await runAssistantTurn(userRecord);
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
function addMessage(messageRecord) {
    const normalizedType = normalizeMessageType(messageRecord.type);
    const timestamp = messageRecord.timestamp ? new Date(messageRecord.timestamp) : new Date();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${normalizedType}`;
    messageDiv.dataset.messageId = messageRecord.id;

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
    messageBody.innerHTML = formatMessageContent(messageRecord.content);
    messageContent.appendChild(messageBody);

    const actions = createMessageActions(messageRecord);
    if (actions) {
        messageContent.appendChild(actions);
    }

    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);

    messagesContainer.appendChild(messageDiv);
    updateMessageBookmarkUI(messageRecord.id, messageRecord.isBookmarked);
    updateMessageReactionUI(messageRecord.id, messageRecord.reaction);
    scrollToBottom();
}

function createMessageActions(messageRecord) {
    const type = normalizeMessageType(messageRecord.type);
    const { id: messageId } = messageRecord;
    let hasActions = false;
    const container = document.createElement('div');
    container.className = 'message-actions';

    if (type === 'user') {
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'message-action-btn';
        editBtn.title = 'Edit message';
        editBtn.textContent = 'Edit';
        editBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            editUserMessage(messageId);
        });
        container.appendChild(editBtn);
        hasActions = true;
    } else if (type === 'assistant') {
        const regenBtn = document.createElement('button');
        regenBtn.type = 'button';
        regenBtn.className = 'message-action-btn';
        regenBtn.title = 'Regenerate response';
        regenBtn.textContent = 'Regenerate';
        regenBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            regenerateAssistantMessage(messageId);
        });
        container.appendChild(regenBtn);
        hasActions = true;

        const reactions = createReactionControls(messageId, messageRecord.reaction);
        container.appendChild(reactions);
    }

    const bookmarkBtn = document.createElement('button');
    bookmarkBtn.type = 'button';
    bookmarkBtn.className = 'message-action-btn bookmark';
    bookmarkBtn.title = 'Bookmark message';
    bookmarkBtn.innerHTML = '☆';
    bookmarkBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        toggleBookmarkMessage(messageId);
    });
    container.appendChild(bookmarkBtn);
    hasActions = true;

    return hasActions ? container : null;
}

function createReactionControls(messageId, currentReaction) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message-reactions';

    const reactions = [
        { icon: '👍', value: 'up', label: 'Helpful' },
        { icon: '👎', value: 'down', label: 'Not helpful' },
    ];

    reactions.forEach((reaction) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'message-action-btn reaction';
        btn.dataset.reaction = reaction.value;
        btn.title = reaction.label;
        btn.textContent = reaction.icon;
        if (reaction.value === currentReaction) {
            btn.classList.add('active');
        }
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            toggleReaction(messageId, reaction.value);
        });
        wrapper.appendChild(btn);
    });

    return wrapper;
}

function toggleReaction(messageId, reactionValue) {
    const chat = getCurrentChat();
    if (!chat) return;

    const message = chat.messages.find((msg) => msg.id === messageId);
    if (!message || message.type !== 'assistant') return;

    message.reaction = message.reaction === reactionValue ? null : reactionValue;
    saveChats();
    updateMessageReactionUI(messageId, message.reaction);
}

function toggleBookmarkMessage(messageId) {
    const chat = getCurrentChat();
    if (!chat) return;

    const message = chat.messages.find((msg) => msg.id === messageId);
    if (!message) return;

    message.isBookmarked = !message.isBookmarked;
    saveChats();
    updateMessageBookmarkUI(messageId, message.isBookmarked);
    refreshBookmarksPanel();
}

function updateMessageBookmarkUI(messageId, isBookmarked) {
    const el = messagesContainer.querySelector(`[data-message-id="${messageId}"]`);
    if (!el) return;

    el.classList.toggle('bookmarked', isBookmarked);
    const action = el.querySelector('.message-action-btn.bookmark');
    if (action) {
        action.innerHTML = isBookmarked ? '★' : '☆';
        action.title = isBookmarked ? 'Remove bookmark' : 'Bookmark message';
    }
}

function updateMessageReactionUI(messageId, reaction) {
    const el = messagesContainer.querySelector(`[data-message-id="${messageId}"]`);
    if (!el) return;
    const buttons = el.querySelectorAll('.message-action-btn.reaction');
    buttons.forEach((btn) => {
        const value = btn.dataset.reaction;
        btn.classList.toggle('active', reaction === value && !!reaction);
    });
}

function refreshBookmarksPanel() {
    if (!bookmarkList || !bookmarkCount) return;
    const chat = getCurrentChat();
    if (!chat) {
        bookmarkList.innerHTML = '<p class="bookmark-empty">No chat selected.</p>';
        bookmarkCount.textContent = '0';
        return;
    }

    const bookmarked = chat.messages.filter((msg) => msg.isBookmarked);
    bookmarkCount.textContent = String(bookmarked.length);

    if (!bookmarked.length) {
        bookmarkList.innerHTML = '<p class="bookmark-empty">No bookmarks yet.</p>';
        return;
    }

    bookmarkList.innerHTML = '';
    bookmarked.forEach((msg) => {
        const item = document.createElement('div');
        item.className = 'bookmark-item';
        item.onclick = () => scrollToMessage(msg.id);
        item.innerHTML = `
            <span class="bookmark-item-title">${truncateText(msg.content, 45)}</span>
            <span class="bookmark-item-meta">${new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        `;
        bookmarkList.appendChild(item);
    });
}

function scrollToMessage(messageId) {
    const el = messagesContainer.querySelector(`[data-message-id="${messageId}"]`);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.classList.add('bookmark-focus');
    setTimeout(() => el.classList.remove('bookmark-focus'), 1500);
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

    if (!messageRecord.id) {
        messageRecord.id = generateMessageId();
    }
    messageRecord.type = normalizeMessageType(messageRecord.type);
    messageRecord.timestamp = messageRecord.timestamp || new Date().toISOString();
    if (typeof messageRecord.isBookmarked !== 'boolean') {
        messageRecord.isBookmarked = Boolean(messageRecord.isBookmarked);
    }
    if (messageRecord.reaction !== 'up' && messageRecord.reaction !== 'down') {
        messageRecord.reaction = null;
    }

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

    const term = chatSearchTerm.trim().toLowerCase();
    const filteredChats = term ? chats.filter((chat) => chatMatchesSearch(chat, term)) : chats;

    if (!filteredChats.length) {
        const emptyMessage = term
            ? `No chats match "${escapeHtml(chatSearchTerm)}".`
            : 'No chats yet.';
        chatList.innerHTML = `<div class="chat-empty">${emptyMessage}</div>`;
        return;
    }

    filteredChats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = `chat-item ${chat.id === currentChatId ? 'active' : ''}`;
        chatItem.onclick = () => loadChat(chat.id);

        const titleHTML = highlightSearchMatch(chat.title || 'Untitled chat');
        const snippet = getChatSnippet(chat);
        const snippetHTML = snippet ? highlightSearchMatch(snippet) : '';

        chatItem.innerHTML = `
            <div class="chat-item-main">
                <div class="chat-item-title">${titleHTML}</div>
                <div class="chat-item-time">${formatTime(chat.lastMessageAt)}</div>
                ${snippetHTML ? `<div class="chat-item-snippet">${snippetHTML}</div>` : ''}
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

async function runAssistantTurn(userRecord) {
    showTypingIndicator();

    const researchRecord = createMessageRecord(
        `Researching “${truncateText(userRecord.content, 120)}”…`,
        'research'
    );
    addMessage(researchRecord);
    updateCurrentChat(researchRecord);

    try {
        const response = await getAIResponse(userRecord.content);
        hideTypingIndicator();
        const assistantRecord = createMessageRecord(response, 'assistant');
        addMessage(assistantRecord);
        updateCurrentChat(assistantRecord);
    } catch (error) {
        hideTypingIndicator();
        const errorRecord = createMessageRecord(
            'Sorry, I encountered an error. Please try again.',
            'system'
        );
        addMessage(errorRecord);
        updateCurrentChat(errorRecord);
        console.error('Error getting AI response:', error);
    }
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
            addMessage(msg);
        });
    }

    updateChatList();
    refreshBookmarksPanel();
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
            id: msg.id || generateMessageId(),
            content: msg.content || '',
            type: normalizeMessageType(msg.type),
            timestamp: msg.timestamp || new Date().toISOString(),
            isBookmarked: Boolean(msg.isBookmarked),
            reaction: msg.reaction || null,
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

function getCurrentChat() {
    return chats.find(c => c.id === currentChatId) || null;
}

function editUserMessage(messageId) {
    if (isTyping) return;
    const chat = getCurrentChat();
    if (!chat) return;

    const index = chat.messages.findIndex((msg) => msg.id === messageId);
    if (index === -1) return;
    const target = chat.messages[index];
    if (target.type !== 'user') return;

    if (index !== chat.messages.length - 1) {
        alert('You can only edit the latest user message.');
        return;
    }

    const originalContent = target.content;
    chat.messages.splice(index);
    chat.lastMessageAt = chat.messages.length
        ? chat.messages[chat.messages.length - 1].timestamp
        : new Date().toISOString();
    saveChats();
    loadChat(chat.id);

    messageInput.value = originalContent;
    adjustTextareaHeight();
    messageInput.focus();
}

async function regenerateAssistantMessage(messageId) {
    if (isTyping) return;
    const chat = getCurrentChat();
    if (!chat) return;

    const index = chat.messages.findIndex((msg) => msg.id === messageId);
    if (index === -1) return;
    const target = chat.messages[index];
    if (target.type !== 'assistant') return;

    if (index !== chat.messages.length - 1) {
        alert('You can only regenerate the most recent assistant reply.');
        return;
    }

    let priorUserIndex = -1;
    for (let i = index - 1; i >= 0; i -= 1) {
        if (chat.messages[i].type === 'user') {
            priorUserIndex = i;
            break;
        }
    }

    if (priorUserIndex === -1) return;
    const userRecord = { ...chat.messages[priorUserIndex] };

    chat.messages.splice(priorUserIndex + 1);
    chat.lastMessageAt = chat.messages.length
        ? chat.messages[chat.messages.length - 1].timestamp
        : new Date().toISOString();
    saveChats();
    loadChat(chat.id);

    await runAssistantTurn(userRecord);
    saveChats();
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
        deleteChat,
        editUserMessage,
        regenerateAssistantMessage,
        runAssistantTurn,
        toggleBookmarkMessage,
        toggleReaction,
        refreshBookmarksPanel
    };
}
function handleChatSearch(event) {
    chatSearchTerm = event.target.value || '';
    updateChatList();
}
