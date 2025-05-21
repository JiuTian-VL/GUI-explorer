<template>
	<a-flex justify="center" align="center" vertical gap="5">
		<div :style="{ border: '1px solid #ccc', borderRadius: '5px' }">
			<img
				:width="width"
				:height="height"
				:style="{ borderRadius: '5px' }"
				:src="imageUrl"
				@error="handleImageError"
			/>
		</div>
		<a-flex justify="space-between" align="center" gap="5" style="width: 100%">
			<a-button
				type="primary"
				flex="1"
				style="width: 50%"
				:disabled="store.runningTask"
				@click="onReset"
				>Reset<!-- 重置--></a-button
			>
			<a-button
				flex="1"
				style="width: 50%"
				:disabled="!store.runningTask"
				@click="onTerminate"
				>Terminate<!-- 终止--></a-button
			>
		</a-flex>
	</a-flex>
</template>
<script setup>
import { useGlobalStore } from '/src/store.js'
import { onMounted, onBeforeUnmount, ref } from 'vue'
const width = ref(216)
const height = ref(480)
const store = useGlobalStore()

//var width=216;
//var height=480;
const imageUrl = ref('http://127.0.0.1:8768/get_screenshot')
let fetchController = null // 用于取消未完成的请求
let timerId = null // 用于定时器
onMounted(() => {
	const fetchImage = async () => {
		try {
			// 创建新的AbortController
			const controller = new AbortController()
			fetchController = controller

			// 添加时间戳参数
			const url = `http://127.0.0.1:8768/get_screenshot?t=${Date.now()}`

			// 使用fetch获取图片
			const response = await fetch(url, {
				signal: controller.signal,
				cache: 'no-cache', // 禁用缓存
			})

			if (!response.ok) throw new Error('Network response was not ok')

			// 转换为Blob
			const blob = await response.blob()

			// 创建新的对象URL
			const newUrl = URL.createObjectURL(blob)

			// 释放旧URL的内存
			if (imageUrl.value) {
				URL.revokeObjectURL(imageUrl.value)
			}

			// 更新图片地址
			imageUrl.value = newUrl
		} catch (error) {
			if (error.name !== 'AbortError') {
				console.error('图片下载失败:', error)
			}
		} finally {
			// 设置下次请求
			timerId = setTimeout(fetchImage, 5)
		}
	}

	timerId = setTimeout(fetchImage, 0) // 立即执行首次请求
})

onBeforeUnmount(() => {
	// 清除定时器和中断请求
	clearTimeout(timerId)
	if (fetchController) {
		fetchController.abort()
	}
	// 释放对象URL内存
	if (imageUrl.value) {
		URL.revokeObjectURL(imageUrl.value)
	}
})
async function fetchWithTimeout(url, options = {}) {
	const { timeout = 300000 } = options // 默认300 秒
	const controller = new AbortController()

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
			throw new Error(`请求超时（${timeout}ms）`)
		}
		throw error
	}
}
const onReset = async () => {
	try {
		store.runningTask = true // 开始加载
		// 发送 POST 请求
		// const response = await fetch('http://127.0.0.1:8768/reset', {
		// 	method: 'POST',
		// 	headers: {
		// 		accept: 'application/json',
		// 		'Content-Type': 'application/json',
		// 	},
		// 	//body: JSON.stringify({ task_goal: taskGoal.value }),
		// })
		const response = await fetchWithTimeout('http://127.0.0.1:8768/reset', {
		  method: 'POST',
		  headers: {
		    accept: 'application/json',
		    'Content-Type': 'application/json',
		  },
		  body: JSON.stringify({ 'msg': 'reset' }),
		  timeout: 30*1000 // 30s
		});

		// 处理响应
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const data = await response.text()
		console.log('Server response:', data)
	} finally {
		store.runningTask = false // 无论成功/失败都关闭
	}
}
const onTerminate = async () => {
	try {
		// 发送 POST 请求
		// const response = await fetch('http://127.0.0.1:8767/stop', {
		// 	method: 'POST',
		// 	headers: {
		// 		accept: 'application/json',
		// 		'Content-Type': 'application/json',
		// 	},
		// 	//body: JSON.stringify({ task_goal: taskGoal.value }),
		// })
		store.controller.abort()
		const response = await fetchWithTimeout('http://127.0.0.1:8768/sent_a_massage3', {
		  method: 'POST',
		  headers: {
		    accept: 'application/json',
		    'Content-Type': 'application/json',
		  },
		  body: JSON.stringify({ 'msg': 'stop' }),
		  timeout: 30*1000 // 30s
		});

		// 处理响应
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const data = await response.text()
		console.log('Server response:', data)
	} finally {
		//store.runningTask = false // 无论成功/失败都关闭
		//store.controller=null
	}
}
const handleImageError = () => {
	console.error('图片加载失败')
	// 可选：回退到默认图或重试逻辑
}
</script>
<style></style>
