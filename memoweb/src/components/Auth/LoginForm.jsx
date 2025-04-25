import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff } from 'react-feather';
import { login } from '../../api/api';
import './AuthForms.css';


const LoginForm = ({ role }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notification, setNotification] = useState(null);
  const navigate = useNavigate();

  const validate = () => {
    const newErrors = {};
    
    if (!formData.username.trim()) {
      newErrors.username = 'Username is required';
    } else if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    setIsSubmitting(true);
    setNotification(null);
    
    try {
      const response = await login({
        username: formData.username,
        password: formData.password
      });
      
      setNotification({
        type: 'success',
        message: 'Login successful! Redirecting...'
      });
      
      // Save token
      localStorage.setItem('token', response.access_token);
      localStorage.setItem('role', response.role); // Save the role too
      
      // Redirect based on role
      
      setTimeout(() => {
        const normalizedRole = role;
        
        console.log('Normalized role:', normalizedRole); // Debug
        
        switch(normalizedRole) {
          case 'family':
            navigate('/family');
            break;
          case 'doctor':
            navigate('/doctor');
            break;
          case 'patient':
            navigate('/patient');
            break;
          default:
            navigate('/');
            console.warn('unknown role');
        }
      }, 1000);
      
      
    } catch (error) {
      let errorMessage = 'Invalid credentials';
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      setNotification({
        type: 'error',
        message: errorMessage
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  

  // Clear notification after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  return (
    <form onSubmit={handleSubmit} className="auth-form">
      <h2>Login as {role === 'family' ? 'Family Member' : role}</h2>
      
      {/* Notification */}
      {notification && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}
      
      {/* Form Error */}
      {errors.form && (
        <div className="error-message">{errors.form}</div>
      )}
      
      <div className={`form-group ${errors.username ? 'error' : ''}`}>
        <label>Username</label>
        <input
          type="text"
          value={formData.username}
          onChange={(e) => setFormData({...formData, username: e.target.value})}
          onBlur={() => validate()}
          disabled={isSubmitting}
        />
        {errors.username && (
          <span className="error-text">{errors.username}</span>
        )}
      </div>

      <div className={`form-group ${errors.password ? 'error' : ''}`}>
        <label>Password</label>
        <div className="password-input-container">
          <input
            type={showPassword ? "text" : "password"}
            value={formData.password}
            onChange={(e) => setFormData({...formData, password: e.target.value})}
            onBlur={() => validate()}
            disabled={isSubmitting}
          />
          <button
            type="button"
            className="password-toggle"
            onClick={() => setShowPassword(!showPassword)}
            aria-label={showPassword ? "Hide password" : "Show password"}
            disabled={isSubmitting}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        </div>
        {errors.password && (
          <span className="error-text">{errors.password}</span>
        )}
      </div>

      <button 
        type="submit" 
        className="submit-button"
        disabled={isSubmitting}
      >
        {isSubmitting ? (
          <span className="button-loader"></span>
        ) : (
          'Login'
        )}
      </button>
    </form>
  );
};

export default LoginForm;