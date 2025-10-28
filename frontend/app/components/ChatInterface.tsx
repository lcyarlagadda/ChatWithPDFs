'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Download, Trash2, MessageSquare } from 'lucide-react'
import { PDFChatService } from '../services/pdfChatService'
import { useToast } from './ToastContainer'

interface ChatInterfaceProps {
  pdfService: PDFChatService
  settings: any
}

interface Message {
  id: string
  role: 'user' | 'bot'
  text: string
  timestamp: Date
  citations?: string[]
  metrics?: any
}

export default function ChatInterface({ pdfService, settings }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const toast = useToast()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      text: input.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setIsStreaming(true)

    try {
      const response = await pdfService.askQuestion(input.trim())
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        text: response.answer,
        timestamp: new Date(),
        citations: response.citations,
        metrics: response.metrics
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error asking question:', error)
      toast.error('Failed to get a response. Please check your backend connection.')
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        text: 'Sorry, I encountered an error while processing your question. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
    }
  }

  const clearConversation = () => {
    setMessages([])
    toast.info('Conversation cleared')
  }

  const exportConversation = () => {
    try {
      const conversationText = messages.map(msg => 
        `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.text}`
      ).join('\n\n')
      
      const blob = new Blob([conversationText], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `conversation_${new Date().toISOString().split('T')[0]}.txt`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success('Conversation exported successfully!')
    } catch (error) {
      console.error('Error exporting conversation:', error)
      toast.error('Failed to export conversation')
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-sm h-[600px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <MessageSquare className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-gray-900">Chat with your PDFs</h2>
        </div>
        <div className="flex items-center space-x-2">
          {messages.length > 0 && (
            <>
              <button
                onClick={exportConversation}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
                title="Export Conversation"
              >
                <Download className="h-4 w-4" />
              </button>
              <button
                onClick={clearConversation}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
                title="Clear Conversation"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>Start a conversation by asking a question about your PDF documents.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`chat-message ${
                  message.role === 'user' ? 'user-message' : 'bot-message'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.text}</p>
                
                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {message.citations.map((citation, index) => (
                      <span key={index} className="citation">
                        {citation}
                      </span>
                    ))}
                  </div>
                )}
                
                {/* Metrics */}
                {message.metrics && settings.showChunks && (
                  <div className="mt-2 text-xs text-gray-500">
                    <p>Retrieval: {message.metrics.retrieval_time?.toFixed(2)}s | 
                       Generation: {message.metrics.generation_time?.toFixed(2)}s | 
                       Tokens: {message.metrics.input_tokens + message.metrics.output_tokens}</p>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="chat-message bot-message">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                <span>Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your PDF documents..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>
    </div>
  )
}

