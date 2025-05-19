import { defineStore } from 'pinia'
export const useGlobalStore = defineStore('global', {
  state: () => ({ runningTask: false ,controller: null,}),
  actions: { toggleRunningTask() { this.runningTask = !this.runningTask } }
})