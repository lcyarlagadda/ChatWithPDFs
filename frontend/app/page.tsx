'use client'

import { useState, useRef, useEffect } from 'react'
import { FileText, Settings } from 'lucide-react'
import ChatInterface from './components/ChatInterface'
import FileUpload from './components/FileUpload'
import SettingsPanel from './components/SettingsPanel'
import { PDFChatService } from './services/pdfChatService'
import { useToast } from './components/ToastContainer'

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    numChunks: 3,
    chunkSize: 500,
    temperature: 0.2,
    maxTokens: 512,
    modelType: 'Mistral-7B',
    retrieverType: 'Hybrid (Dense + Sparse)',
    enableNoiseFiltering: true,
    showChunks: false
  })

  const pdfService = useRef(new PDFChatService())
  const toast = useToast()

  const handleFileUpload = async (files: File[]) => {
    setIsProcessing(true)
    try {
      await pdfService.current.uploadAndProcessPDFs(files)
      setIsReady(true)
      toast.success('PDFs processed successfully! You can now start chatting.')
    } catch (error) {
      console.error('Error processing PDFs:', error)
      toast.error('Error processing PDFs. Please check your backend connection and try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleSettingsChange = (newSettings: typeof settings) => {
    setSettings(newSettings)
    pdfService.current.updateSettings(newSettings)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <FileText className="h-8 w-8 text-primary-600" />
              <h1 className="text-xl font-semibold text-gray-900">PDF Chat System</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
                title="Settings"
              >
                <Settings className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Settings Panel */}
          {showSettings && (
            <div className="lg:col-span-1">
              <SettingsPanel
                settings={settings}
                onSettingsChange={handleSettingsChange}
              />
            </div>
          )}

          {/* Main Chat Area */}
          <div className={showSettings ? "lg:col-span-3" : "lg:col-span-4"}>
            {!isReady ? (
              <div className="bg-white rounded-lg shadow-sm p-8">
                <div className="text-center">
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    Upload PDF Documents
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Upload one or more PDF files to start chatting with your documents
                  </p>
                  <FileUpload
                    onFileUpload={handleFileUpload}
                    isProcessing={isProcessing}
                  />
                </div>
              </div>
            ) : (
              <ChatInterface
                pdfService={pdfService.current}
                settings={settings}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

