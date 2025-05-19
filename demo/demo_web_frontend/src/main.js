import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

import Antd from 'ant-design-vue';
import 'ant-design-vue/dist/reset.css';

import { createPinia } from 'pinia'

const app = createApp(App)

app.use(Antd)
app.use(router)

const pinia = createPinia()
app.use(pinia)

app.mount('#app')
