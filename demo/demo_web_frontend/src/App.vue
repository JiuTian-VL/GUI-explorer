<script setup>
import ScreenPage from '/src/components/ScreenPage.vue'
import TaskPage from '/src/components/TaskPage.vue'
import ReasoningPage from '/src/components/ReasoningPage.vue'
import { ref, onMounted, onUnmounted } from 'vue'
import { useGlobalStore } from '/src/store.js'

const store = useGlobalStore()
// 响应式聊天数据
const chatMessages = ref([])

// 添加新消息
//     chatMessages.value.push({
//       messageType: 'text',
//       name: 'System',
//       position: 'left',
//       html: '新消息已送达'
//     })

//  chatMessages.value.push({
//   messageType: 'raw',
//   name: 'Agent',
//   position: 'left',
//   html: `<img
//          src="https://xj-psd-1258344703.cos.ap-guangzhou.myqcloud.com/image/hunyuan/file/url.svg"
//          style="max-width: 200px; border-radius: 8px; margin: 4px 0;"
//        ><br>图片带文字是可以的`
// })

const pollingInterval = ref(null)

const checkMessages = async () => {
	try {
		const response = await fetch('http://127.0.0.1:5768/get_a_massage2', {
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
			//if (jsonData.action) {
			if (data.message_type === 'knowledge') {
				store.runningTask = true
				chatMessages.value.push({
					//messageType: 'text',
					messageType: 'raw',
					name: 'Agent',
					position: 'left',
					//html: jsonData.action
					//html:data.task_goal
					html: `<strong>Dynamic Guidance</strong><br>${data.message}`.replace('\n','<br>'),
					// html: `<strong>知识库检索结果</strong><br>${data.message}`.replace('\n','<br>'),//#TODO: 汉化
				})
			} else if (data.message_type === 'reasoning') {
				store.runningTask = true
				chatMessages.value.push({
					//messageType: 'text',
					messageType: 'raw',
					name: 'Agent',
					position: 'left',
					//html: jsonData.action
					//html:data.task_goal
					html: `<strong>Reasoning</strong><br>${data.message}`.replace('\n', '<br>'),
					// html: `<strong>思考</strong><br>${data.message}`.replace('\n', '<br>'),//#TODO: 汉化
				})
			} else if (data.message_type === 'done') {
				store.runningTask = false
			}/*else {
				chatMessages.value.push({
					//messageType: 'text',
					messageType: 'raw',
					name: 'Agent',
					position: 'left',
					//html: jsonData.action
					//html:data.task_goal
					html: `<p><strong>Unknown</strong><br>${data.message}</p>`,
				})
			}*/
			//}
		} catch (e) {
			console.error('JSON解析失败:', e)
		}
	} catch (error) {
		console.error('获取消息失败:', error)
	}
}

// 生命周期钩子
onMounted(() => {
	pollingInterval.value = setInterval(checkMessages, 499)
})

onUnmounted(() => {
	clearInterval(pollingInterval.value)
})
</script>

<template>
	<div class="main-container">
		<img alt="JiuTian" class="head_logo" src="@/assets/head.png" />
		<div wrap justify="center" align="center" gap="5" class="content-flex">
			<ScreenPage></ScreenPage>
			<div class="task-wrapper">
				<TaskPage></TaskPage>
			</div>
			<div class="reasoning-wrapper">
				<ReasoningPage :messages="chatMessages" class="reasoning-content"></ReasoningPage>
			</div>
		</div>
	</div>
</template>

<style scoped>
/* 在全局样式或父容器中 */
body,
#app {
	height: 100%; /* 确保html/body高度占满视口 */
	min-height: 100vh; /* 兼容移动端 */
	margin: 10px 10px 10px 10px;
	display: flex;
	justify-content: center; /* 水平居中 */
	align-items: center; /* 垂直居中 */
}
.main-container {
	min-height: 100vh; /* 关键：占据完整视口高度 */
	display: flex;
	flex-direction: column; /* 上下布局 */
	gap: 5px;
	align-items: center;
}
.head_logo {
	width: 658px;
	height: auto;
}
/* Flex 容器自适应 */
.content-flex {
	/*width: 100%;
	height: 100%;*/
	display: flex;
	gap: 5px;
	align-items: center;
	width: 658px;
}

/* ReasoningPage 固定宽度容器 */
.reasoning-wrapper {
	background: #f8f9fa;
	border: 0.5px solid #ccc;
	border-radius: 5px;
}

/* ReasoningPage 内容尺寸 */
.reasoning-content {
	width: 216px;
	height: 515px;
}
</style>
