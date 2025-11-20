'use client'

import { Settings } from 'lucide-react'

interface SettingsPanelProps {
  settings: {
    numChunks: number
    chunkSize: number
    temperature: number
    maxTokens: number
    modelType: string
    retrieverType: string
    enableNoiseFiltering: boolean
    showChunks: boolean
  }
  onSettingsChange: (settings: any) => void
}

export default function SettingsPanel({ settings, onSettingsChange }: SettingsPanelProps) {
  const handleChange = (key: string, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    })
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <Settings className="h-5 w-5 mr-2" />
          Settings
        </h3>
      </div>

      <div className="space-y-6">
        {/* Model Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Model Type
          </label>
          <select
            value={settings.modelType}
            onChange={(e) => handleChange('modelType', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="Mistral-7B">Mistral-7B</option>
            <option value="FLAN-T5">FLAN-T5</option>
            <option value="GPT-2">GPT-2</option>
          </select>
        </div>

        {/* Retriever Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Retriever Type
          </label>
          <select
            value={settings.retrieverType}
            onChange={(e) => handleChange('retrieverType', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="Hybrid (Dense + Sparse)">Hybrid (Dense + Sparse)</option>
            <option value="Dense Only">Dense Only</option>
            <option value="Sparse Only">Sparse Only</option>
            <option value="LlamaIndex">LlamaIndex</option>
          </select>
        </div>

        {/* Number of Chunks */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Number of Chunks: {settings.numChunks}
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={settings.numChunks}
            onChange={(e) => handleChange('numChunks', parseInt(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            More chunks = more context but slower
          </p>
        </div>

        {/* Chunk Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Chunk Size: {settings.chunkSize}
          </label>
          <input
            type="range"
            min="200"
            max="1000"
            step="50"
            value={settings.chunkSize}
            onChange={(e) => handleChange('chunkSize', parseInt(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Size of text chunks for processing
          </p>
        </div>

        {/* Temperature */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Temperature: {settings.temperature}
          </label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Controls randomness in responses
          </p>
        </div>

        {/* Max Tokens */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Tokens: {settings.maxTokens}
          </label>
          <input
            type="range"
            min="100"
            max="1000"
            step="50"
            value={settings.maxTokens}
            onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Maximum tokens in response
          </p>
        </div>

        {/* Checkboxes */}
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={settings.enableNoiseFiltering}
              onChange={(e) => handleChange('enableNoiseFiltering', e.target.checked)}
              className="rounded border-gray-300 text-primary-600 shadow-sm focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Enable Advanced Noise Filtering
            </span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={settings.showChunks}
              onChange={(e) => handleChange('showChunks', e.target.checked)}
              className="rounded border-gray-300 text-primary-600 shadow-sm focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Show Retrieved Chunks (Debug)
            </span>
          </label>
        </div>
      </div>
    </div>
  )
}

