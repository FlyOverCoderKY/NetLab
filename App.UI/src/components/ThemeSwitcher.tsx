import { Menu } from '@headlessui/react'
import { useTheme } from '../context/ThemeContext'
import './ThemeSwitcher.css'

interface ThemeSwitcherProps {
  appearance: 'light' | 'dark' | 'system'
  onAppearanceChange: (appearance: 'light' | 'dark' | 'system') => void
}

export function ThemeSwitcher({ appearance, onAppearanceChange }: ThemeSwitcherProps) {
  useTheme()

  const getAppearanceIcon = () => {
    switch (appearance) {
      case 'light':
        return '☀️'
      case 'dark':
        return '🌙'
      case 'system':
        return '🖥️'
      default:
        return '☀️'
    }
  }

  return (
    <Menu as="div" className="dropdown-container">
      <Menu.Button className="dropdown-trigger">
        {getAppearanceIcon()}
        <span style={{ marginLeft: 4, marginRight: 4 }}>Theme</span>
        <span>▼</span>
      </Menu.Button>

      <Menu.Items className="dropdown-items">
        <Menu.Item>
          {({ active }) => (
            <button
              className={`dropdown-item ${active ? 'dropdown-item-active' : ''}`}
              onClick={() => onAppearanceChange('light')}
            >
              ☀️ Light
            </button>
          )}
        </Menu.Item>

        <Menu.Item>
          {({ active }) => (
            <button
              className={`dropdown-item ${active ? 'dropdown-item-active' : ''}`}
              onClick={() => onAppearanceChange('dark')}
            >
              🌙 Dark
            </button>
          )}
        </Menu.Item>

        <Menu.Item>
          {({ active }) => (
            <button
              className={`dropdown-item ${active ? 'dropdown-item-active' : ''}`}
              onClick={() => onAppearanceChange('system')}
            >
              🖥️ System
            </button>
          )}
        </Menu.Item>
      </Menu.Items>
    </Menu>
  )
}


