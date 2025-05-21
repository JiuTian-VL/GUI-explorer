<template>
	<div class="chat-container">
		<div ref="chatbox" class="lite-chatbox"></div>
	</div>
</template>

<script>
const TitleType = { admin: 'admin', owner: 'owner' }
var keepScrolledToBottom=true

function beforeRenderingHTML(messages, selector) {
	const container = document.querySelector(selector)
	if (!container) return

	let html = ''
	messages.forEach((msg) => {
		html += `
      <div class="c${msg.position} cmsg">
        <span class="name">
          ${renderTitleHtml(msg.htitle, TitleType[msg.htitleType] || '')}
          <span>${escapeHtml(msg.name) || '&nbsp;'}</span>
        </span>
        <span class="content">${msg.messageType === 'raw' ? msg.html : escapeHtml(msg.html)}</span>
      </div>
    `
	})

	container.innerHTML = html
	if (keepScrolledToBottom) {
		container.scrollTop = container.scrollHeight
		const scrollToBottom = () => {
			container.scrollTo({
				top: container.scrollHeight,
				behavior: 'smooth', // 启用平滑滚动
			})
		}
		const images = container.querySelectorAll('img')
		if (images.length > 0) {
			let loadedCount = 0
			images.forEach((img) => {
				if (img.complete) {
					loadedCount++
				} else {
					img.onload = () => {
						loadedCount++
						if (loadedCount === images.length) scrollToBottom()
					}
				}
			})
			if (loadedCount === images.length) scrollToBottom()
		}
	}
}

function renderTitleHtml(htitle, type) {
	return htitle ? `<span class="htitle ${type}" style="margin: 0 4px 0 0;">${htitle}</span>` : ''
}

function escapeHtml(text) {
	const div = document.createElement('div')
	div.textContent = text
	return div.innerHTML
}

export default {
	name: 'ChatComponent',
	props: {
		messages: {
			type: Array,
			required: true,
			default: () => [],
		},
	},
	data() {
		return {
			scrollDebounce: null, // 滚动节流
			htmls: [],
		}
	},
	watch: {
		messages: {
			deep: true,
			immediate: true,
			handler(newVal) {
				this.htmls = [...newVal]
				this.$nextTick(this.renderChat)
			},
		},
	},
	mounted() {
		this.renderChat()
		this.$nextTick(() => {
			const container = this.$refs.chatbox
			container.addEventListener('scroll', this.handleScroll)
		})
	},
	beforeUnmount() {
		const container = this.$refs.chatbox
		container.removeEventListener('scroll', this.handleScroll)
	},
	methods: {
		renderChat() {
			if (typeof beforeRenderingHTML === 'function') {
				this.$nextTick(() => {
					beforeRenderingHTML(this.htmls, '.lite-chatbox')
				})
			}
		},
		handleScroll() {
			const container = this.$refs.chatbox
			const threshold = 100 // 滚动阈值

			// 节流处理（300ms）
			if (this.scrollDebounce) return
			if (container.scrollHeight > 2 * container.clientHeight)
				this.scrollDebounce = setTimeout(() => {
					const isNearBottom =
						container.scrollHeight - container.scrollTop - container.clientHeight <=
						threshold

					keepScrolledToBottom = isNearBottom
					this.scrollDebounce = null
				}, 300)
		},
	},
}
</script>

<style scoped>
html {
	line-height: 1;
	-webkit-text-size-adjust: 100%;
}
body {
	margin: 0;
}
main {
	display: block;
}
h1 {
	font-size: 2em;
	margin: 0.67em 0;
}
hr {
	-webkit-box-sizing: content-box;
	box-sizing: content-box;
	height: 0;
	overflow: visible;
}
pre {
	font-family: monospace, monospace;
	font-size: 1em;
}
a {
	background-color: transparent;
}
abbr[title] {
	border-bottom: none;
	text-decoration: underline;
	-webkit-text-decoration: underline dotted;
	text-decoration: underline dotted;
}
b,
strong {
	font-weight: bolder;
}
code,
kbd,
samp {
	font-family: monospace, monospace;
	font-size: 1em;
}
small {
	font-size: 80%;
}
sub,
sup {
	font-size: 75%;
	line-height: 0;
	position: relative;
	vertical-align: baseline;
}
sub {
	bottom: -0.25em;
}
sup {
	top: -0.5em;
}
img {
	border-style: none;
}
button,
input,
optgroup,
select,
textarea {
	font-family: inherit;
	font-size: 100%;
	line-height: 1;
	margin: 0;
}
button,
input {
	overflow: visible;
}
button,
select {
	text-transform: none;
}
[type='button'],
[type='reset'],
[type='submit'],
button {
	-webkit-appearance: button;
	appearance: button;
}
[type='button']::-moz-focus-inner,
[type='reset']::-moz-focus-inner,
[type='submit']::-moz-focus-inner,
button::-moz-focus-inner {
	border-style: none;
	padding: 0;
}
[type='button']:-moz-focusring,
[type='reset']:-moz-focusring,
[type='submit']:-moz-focusring,
button:-moz-focusring {
	outline: 1px dotted ButtonText;
}
fieldset {
	padding: 0.35em 0.75em 0.625em;
}
legend {
	-webkit-box-sizing: border-box;
	box-sizing: border-box;
	color: inherit;
	display: table;
	max-width: 100%;
	padding: 0;
	white-space: normal;
}
progress {
	vertical-align: baseline;
}
textarea {
	overflow: auto;
}
[type='checkbox'],
[type='radio'] {
	-webkit-box-sizing: border-box;
	box-sizing: border-box;
	padding: 0;
}
[type='number']::-webkit-inner-spin-button,
[type='number']::-webkit-outer-spin-button {
	height: auto;
}
[type='search'] {
	-webkit-appearance: textfield;
	appearance: textfield;
	outline-offset: -2px;
}
[type='search']::-webkit-search-decoration {
	-webkit-appearance: none;
}
::-webkit-file-upload-button {
	-webkit-appearance: button;
	font: inherit;
}
details {
	display: block;
}
summary {
	display: list-item;
}
template {
	display: none;
}
[hidden] {
	display: none;
}
* {
	scrollbar-color: #5c6163 rgba(56, 59, 60, 0.031372549);
}
::-webkit-scrollbar {
	width: 7px;
	height: 1px;
}
::-webkit-scrollbar-thumb {
	border-radius: 10px;
	background-color: rgba(144, 147, 153, 0.5);
	border: 0;
}
[litewebchat-theme='dark'] ::-webkit-scrollbar-thumb {
	background-color: rgba(84, 91, 95, 0.5);
}
::-webkit-scrollbar-track {
	background: #fff;
	min-height: 50%;
	min-height: 20px;
}
[litewebchat-theme='dark'] ::-webkit-scrollbar-track {
	background: #181a1b;
}
::-webkit-scrollbar-corner {
	background-color: transparent;
}
::-moz-selection {
	background-color: #1963bd !important;
	color: #f8f6f3 !important;
}
::selection {
	background-color: #1963bd !important;
	color: #f8f6f3 !important;
}
body {
	font-family: Helvetica, 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
:deep(.lite-chatbox) {
	scroll-behavior: smooth;
	overscroll-behavior: contain;
	padding: 0;
	width: 100%;
	position: relative;
	font-size: 14px;
	/*background-color: #f8f9fa;*/
	overflow-y: auto;
	overflow-x: hidden;
	max-height: 100%; /* 确保内容超出时触发滚动 */
}
[litewebchat-theme='dark'] :deep(.lite-chatbox) {
	background-color: #131415;
}
:deep(.lite-chatbox .cmsg) {
	position: relative;
	margin: 4px 4px;
	min-height: 50px;
	border: 0;
}
:deep(.lite-chatbox .cright) {
	text-align: right;
	margin-left: 4px;
}

:deep(.lite-chatbox .cright .name) {
	margin: 0 0px 2px 0;
}
:deep(.lite-chatbox .cright .content) {
	margin: 0 0px 0 0;
	border-radius: 10px 0 10px 10px;
	color: #fff;
	text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
	background: -o-linear-gradient(70deg, rgba(63, 143, 225, 0.8) 0, #44d7c9 100%);
	background: linear-gradient(20deg, rgba(63, 143, 225, 0.8) 0, #44d7c9 100%);
	-webkit-box-shadow: 5px 5px 15px 0 rgba(102, 102, 102, 0.15);
	box-shadow: 5px 5px 15px 0 rgba(102, 102, 102, 0.15);
}
[litewebchat-theme='dark'] :deep(.lite-chatbox .cright .content) {
	background: -o-linear-gradient(70deg, rgba(25, 91, 159, 0.8) 0, #219a92 100%);
	background: linear-gradient(20deg, rgba(25, 91, 159, 0.8) 0, #219a92 100%);
}
:deep(.lite-chatbox .cright .content::after) {
	left: -12px;
	top: 8px;
}
:deep(.lite-chatbox .cleft) {
	text-align: left;
	margin-right: 4px;
}

:deep(.lite-chatbox .cleft .name) {
	margin: 0 0 2px 0px;
}
:deep(.lite-chatbox .cleft .content) {
	margin: 0 0 0 0px;
	border-radius: 0 10px 10px 10px;
	background: #fff;
	color: #373737;
	border: 1px solid rgba(0, 0, 0, 0.05);
	-webkit-box-shadow: 5px 5px 15px 0 rgba(102, 102, 102, 0.1);
	box-shadow: 5px 5px 15px 0 rgba(102, 102, 102, 0.1);
}
[litewebchat-theme='dark'] :deep(.lite-chatbox .cleft .content) {
	background: #22242a;
}
[litewebchat-theme='dark'] :deep(.lite-chatbox .cleft .content) {
	color: #d4d4d4;
}
:deep(.lite-chatbox .cleft .content::after) {
	left: -12px;
	top: 8px;
}

:deep(.lite-chatbox img.radius) {
	border-radius: 50%;
}
:deep(.lite-chatbox .name) {
	color: #8b8b8b;
	font-size: 14px;
	display: block;
	line-height: 16px;
}
:deep(.lite-chatbox .name > span) {
	vertical-align: middle;
}
:deep(.lite-chatbox .name .htitle) {
	display: inline-block;
	padding: 0 3px 0 3px;
	background-color: #ccc;
	color: #fff;
	border-radius: 4px;
	margin-right: 4px;
	font-size: 14px;
	overflow: hidden;
	-o-text-overflow: ellipsis;
	text-overflow: ellipsis;
	white-space: nowrap;
	vertical-align: middle;
	max-width: 50px;
}
[litewebchat-theme='dark'] :deep(.lite-chatbox .name .htitle) {
	background-color: #4c5052;
}
:deep(.lite-chatbox .name .htitle.admin) {
	background-color: #72d6a0;
}
[litewebchat-theme='dark'] :deep(.lite-chatbox .name .htitle.admin) {
	background-color: #3c916e;
}
:deep(.lite-chatbox .name .htitle.owner) {
	background-color: #f2bf25;
}
[litewebchat-theme='dark'] :deep(.lite-chatbox .name .htitle.owner) {
	background-color: #9a7c21;
}
:deep(.lite-chatbox .content) {
	word-break: break-all;
	word-wrap: break-word;
	text-align: left;
	/*text-align: center;*/
	position: relative;
	display: inline-block;
	font-size: 14px;
	padding: 4px 4px; /*文本框和文字之间的空白*/
	line-height: 18px;
	white-space: pre-wrap;
	width: 100%;
	min-height: 18px;
}
:deep(.lite-chatbox .content img) {
	width: 100%;
	height: auto;
}
:deep(.lite-chatbox .content a) {
	color: #0072c1;
	margin: 0 5px;
	cursor: hand;
}
:deep([litewebchat-theme='dark'] .lite-chatbox .content a) {
	color: #00c3ff;
}
</style>
