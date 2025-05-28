<template>
	<a-flex justify="center" align="center" vertical gap="5">
		<div class="chat-wrapper">
			<ChatPage :messages="chatMessages" ref="chatComp" class="chat-container" />
		</div>
		<a-input-group compact style="display: flex">
			<a-auto-complete
				style="flex: 1"
				v-model:value="taskGoal"
				:options="options"
				allow-clear
				:disabled="store.runningTask"
			/>
			<a-button type="primary" :loading="store.runningTask" @click="handleSearch">Run<!-- 执行 --></a-button>
		</a-input-group>
	</a-flex>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import ChatPage from '/src/components/ChatPage.vue'
import { useGlobalStore } from '/src/store.js'

const store = useGlobalStore()
const taskGoal = ref('')
const options = ref([
	{
		value: `Add the following recipe into the Broccoli app:
Title: Spicy Tuna Wraps
Description: A quick and easy meal, perfect for busy weekdays.`,
	},
	{
		value: `Add the following recipe into the Broccoli app:
Title: Greek Salad Pita Pockets
Description: An ideal recipe for experimenting with different flavors and ingredients.
Servings: 3-4 servings
Preparation Time: 20 mins
Ingredients: various amounts
Directions: Fill pita pockets with lettuce, cucumber, tomato, feta, olives, and Greek dressing. Feel free to substitute with ingredients you have on hand.`,
	},
])

// 响应式聊天数据
const chatMessages = ref([])

// 添加新消息
chatMessages.value.push({
	messageType: 'text',
	name: 'Agent',
	position: 'left',
	html: 'Please send me a task goal.',
	// html: '请给我发一个任务目标。',//#TODO: 汉化
})
async function fetchWithTimeout(url, options = {}, controller = new AbortController()) {
	const { timeout = 300000 } = options // 默认300 秒

	const timeoutId = setTimeout(() => {
		controller.abort() // 强制终止请求
	}, timeout)

	try {
		const response = await fetch(url, {
			...options,
			signal: controller.signal,
		})
		clearTimeout(timeoutId)
		return response
	} catch (error) {
		if (error.name === 'AbortError') {
			throw new Error(`请求超时或被取消（${timeout}ms）`)
		}
		throw error
	}
}
const handleSearch = async () => {
	try {
		store.runningTask = true

		chatMessages.value.push({
			messageType: 'text',
			name: 'Human',
			position: 'right',
			html: taskGoal.value,
		})

		store.controller = new AbortController()
		const response = await fetchWithTimeout(
			'http://127.0.0.1:8767/run_task',
			{
				method: 'POST',
				headers: {
					accept: 'application/json',
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ task_goal: taskGoal.value }),
				timeout: 3000 * 1000, // 3000s
			},
			store.controller
		)

		// 处理响应
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const data = await response.text()
		console.log('Server response:', data)
	} catch (error) {
		console.error('请求失败:', error)
		// 可以在这里添加错误消息到聊天窗口
		chatMessages.value.push({
			messageType: 'text',
			name: 'System',
			position: 'left',
			html: `请求失败: ${error.message}`,
		})
	} finally {
		store.controller = null
		store.runningTask = false
	}
}

const pollingInterval = ref(null)

const checkMessages = async () => {
	try {
		const response = await fetch('http://127.0.0.1:8768/get_a_massage', {
			method: 'GET',
			headers: {
				accept: 'application/json',
				'Content-Type': 'application/json',
			},
		})

		if (!response.ok) return
		const data = await response.json()

		if (data === '<None>') return

		//if ("<None>" in data) return

		try {
			//const jsonData = JSON.parse(data)
			if (data.message_type === 'action') {
				store.runningTask = true
				chatMessages.value.push({
					//messageType: 'text',
					messageType: 'raw',
					name: 'Agent',
					position: 'left',
					//html: jsonData.action
					//html:data.task_goal
					html: `<strong>Action</strong><br>${data.message}`.replace('\n', '<br>'),
					// html: `<strong>行动</strong><br>${data.message}`.replace('\n', '<br>'),//#TODO: 汉化
				})
			} else if (data.message_type === 'summary') {
				store.runningTask = true
				chatMessages.value.push({
					//messageType: 'text',
					messageType: 'raw',
					name: 'Agent',
					position: 'left',
					//html: jsonData.action
					//html:data.task_goal
					html: `<strong>Summary</strong><br>${data.message}`.replace('\n', '<br>'),
					// html: `<strong>总结</strong><br>${data.message}`.replace('\n', '<br>'),//#TODO: 汉化
				})
			} else if (data.message_type === 'done') {
				store.runningTask = false
			}
		} catch (e) {
			console.error('JSON解析失败:', e)
		}
	} catch (error) {
		console.error('获取消息失败:', error)
	}
}

// 生命周期钩子
onMounted(() => {
	pollingInterval.value = setInterval(checkMessages, 500)
})

onUnmounted(() => {
	clearInterval(pollingInterval.value)
})
</script>

<style scoped>
.chat-wrapper {
	border: 0.5px solid #ccc;
	border-radius: 5px;
	background-color: #f8f9fa;
	height: 480px;
	width: 216px;
	/*width: 100%;
	height: 100%;*/
}
.chat-container {
	width: 100%;
	height: 100%;
}
</style>
