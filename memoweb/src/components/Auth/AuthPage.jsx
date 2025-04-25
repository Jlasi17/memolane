import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import RoleSelection from './RoleSelection';
import LoginForm from './LoginForm';
import SignupForm from './SignupForm';
import { Moon, Sun } from 'react-feather';
import './AuthPage.css';

const AuthPage = () => {
  const [authMode, setAuthMode] = useState('login');
  const [selectedRole, setSelectedRole] = useState(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    document.body.className = darkMode ? 'dark' : 'light';
  }, [darkMode]);

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
    if (role === 'patient') setAuthMode('login');
  };

  return (
    <div className={`auth-container ${darkMode ? 'dark' : 'light'}`}>
      <button 
        className="theme-toggle"
        onClick={() => setDarkMode(!darkMode)}
        aria-label="Toggle dark mode"
      >
        {darkMode ? <Sun size={20} /> : <Moon size={20} />}
      </button>

      <div className="auth-card">
        <h1 className="app-title">Memory Lane</h1>
        
        {!selectedRole ? (
          <motion.div
            key="role-selection"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <RoleSelection onRoleSelect={handleRoleSelect} />
          </motion.div>
        ) : (
          <div className="auth-forms-container">
            <button 
              className="back-button"
              onClick={() => setSelectedRole(null)}
            >
              ← Back
            </button>
            
            <div className="auth-header">
              <button 
                className={`tab-button ${authMode === 'login' ? 'active' : ''}`}
                onClick={() => setAuthMode('login')}
              >
                Login
              </button>
              {selectedRole !== 'patient' && (
                <button 
                  className={`tab-button ${authMode === 'signup' ? 'active' : ''}`}
                  onClick={() => setAuthMode('signup')}
                >
                  Sign Up
                </button>
              )}
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={authMode}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {authMode === 'login' ? (
                  <LoginForm role={selectedRole} />
                ) : (
                  <SignupForm role={selectedRole} />
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
};

export default AuthPage;