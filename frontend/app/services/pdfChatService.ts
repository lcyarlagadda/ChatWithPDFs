import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export class PDFChatService {
  private settings: any = {}

  constructor() {
    // Initialize with default settings
    this.settings = {
      numChunks: 3,
      chunkSize: 500,
      temperature: 0.2,
      maxTokens: 512,
      modelType: 'Mistral-7B',
      retrieverType: 'Hybrid (Dense + Sparse)',
      enableNoiseFiltering: true,
      showChunks: false
    }
  }

  async uploadAndProcessPDFs(files: File[]): Promise<void> {
    const formData = new FormData()
    
    files.forEach((file, index) => {
      formData.append(`files`, file)
    })

    // Add settings to form data
    Object.entries(this.settings).forEach(([key, value]) => {
      formData.append(key, value.toString())
    })

    try {
      const response = await axios.post(`${API_BASE_URL}/upload-pdfs`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout for processing
      })

      if (response.status !== 200) {
        throw new Error('Failed to process PDFs')
      }

      return response.data
    } catch (error) {
      console.error('Error uploading PDFs:', error)
      throw error
    }
  }

  async askQuestion(question: string): Promise<any> {
    try {
      const response = await axios.post(`${API_BASE_URL}/ask-question`, {
        question,
        settings: this.settings
      }, {
        timeout: 120000, // 2 minutes timeout for questions
      })

      return response.data
    } catch (error) {
      console.error('Error asking question:', error)
      throw error
    }
  }

  async getHealth(): Promise<boolean> {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000
      })
      return response.status === 200
    } catch (error) {
      console.error('Health check failed:', error)
      return false
    }
  }

  updateSettings(newSettings: any): void {
    this.settings = { ...this.settings, ...newSettings }
  }

  getSettings(): any {
    return this.settings
  }
}

