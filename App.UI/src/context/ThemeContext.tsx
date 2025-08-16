import { createContext, useContext, useEffect, useState } from 'react'

export interface ThemeContextType {
  appearance: 'light' | 'dark' | 'system'
  setAppearance: (appearance: 'light' | 'dark' | 'system') => void
  resolvedAppearance: 'light' | 'dark'
}

const ThemeContext = createContext<ThemeContextType | null>(null)

const THEME_STORAGE_KEY = 'template-theme-preferences'

type ThemePreferences = {
  appearance: 'light' | 'dark' | 'system'
}

interface ThemeProviderProps {
  children: (theme: ThemeContextType) => React.ReactNode
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [preferences, setPreferences] = useState<ThemePreferences>(() => {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed && typeof parsed === 'object' && ['light', 'dark', 'system'].includes(parsed.appearance)) {
          return parsed
        }
      }
    } catch {
      // ignore
    }
    return { appearance: 'system' }
  })

  const [resolvedAppearance, setResolvedAppearance] = useState<'light' | 'dark'>(() => {
    try {
      if (preferences.appearance === 'system') {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
      }
      return preferences.appearance
    } catch {
      return 'light'
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(preferences))
    } catch {
      // ignore
    }
  }, [preferences])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', resolvedAppearance)
  }, [resolvedAppearance])

  useEffect(() => {
    if (preferences.appearance !== 'system') return
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = (e: MediaQueryListEvent) => {
      const newAppearance = e.matches ? 'dark' : 'light'
      setResolvedAppearance(newAppearance)
      document.documentElement.setAttribute('data-theme', newAppearance)
    }
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [preferences.appearance])

  const value: ThemeContextType = {
    appearance: preferences.appearance,
    setAppearance: (appearance: 'light' | 'dark' | 'system') => {
      setPreferences((prev: ThemePreferences) => ({ ...prev, appearance }))
      if (appearance !== 'system') {
        setResolvedAppearance(appearance)
      } else {
        const systemAppearance = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
        setResolvedAppearance(systemAppearance)
      }
    },
    resolvedAppearance,
  }

  return <ThemeContext.Provider value={value}>{children(value)}</ThemeContext.Provider>
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    return {
      appearance: 'light' as const,
      setAppearance: () => {},
      resolvedAppearance: 'light' as const,
    }
  }
  return context
}


