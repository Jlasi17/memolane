
import { useState, useEffect } from 'react';
import {
  PatientRegistrationForm,
  MRIScanUpload,
  ImageUpload,
  PatientStats,
  Notifications
} from '../index';
import './famstyles.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const FamilyHome = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [patientData, setPatientData] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [registrationType, setRegistrationType] = useState('new');
  const [existingPatientId, setExistingPatientId] = useState('');
  const [showMRIModal, setShowMRIModal] = useState(false);
  const [showImageModal, setShowImageModal] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [username, setUsername] = useState('');
  const [showNotifications, setShowNotifications] = useState(false); // Add this to your existing state


  // Check if patient exists on component mount
  useEffect(() => {
    const checkPatient = async () => {
      try {
        await fetchPatientData();
        await fetchNotifications();
        await fetchUsername();
      } catch (error) {
        console.log('No patient data found');
      }
    };
    checkPatient();
  }, []);

  const fetchPatientData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/patient_stats`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        if (response.status === 404) {
          setActiveTab('register');
          throw new Error('No patient data found');
        }
        throw new Error(`Failed to fetch patient data: ${response.status}`);
      }

      const data = await response.json();
      setPatientData(data.patient);
      setStats(data.stats);
      setActiveTab('dashboard');
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchNotifications = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/notifications`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setNotifications(data);
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  };  

  const fetchUsername = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/user`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        // If the endpoint doesn't exist or returns an error
        throw new Error('User info not available');
      }
      
      const data = await response.json();
      setUsername(data.username || data.email || 'User');
    } catch (error) {
      console.error('Error fetching username:', error);
      // Fallback to getting email from token or using a default
      const token = localStorage.getItem('token');
      if (token) {
        try {
          const payload = JSON.parse(atob(token.split('.')[1]));
          setUsername(payload.email || payload.sub || 'User');
        } catch (e) {
          setUsername('User');
        }
      } else {
        setUsername('User');
      }
    }
  };
  const handlePatientRegistration = async (formData) => {
    setIsLoading(true);
    setError(null); // Clear previous errors
    
    try {
      if (registrationType === 'existing') {
        // Validate existing patient ID
        if (!existingPatientId.trim()) {
          throw new Error('Patient ID is required');
        }
  
        const response = await fetch(`${API_BASE_URL}/api/register_patient`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          body: JSON.stringify({
            patient_id: existingPatientId,
            name: "Linked Patient", // Required by backend model
            age: 65, // Required by backend model
            gender: "other", // Required by backend model
            email: `replay.test@gmail.com`, // Required by backend model
            phone: "0000000000", // Required by backend model
            // These fields will be ignored for existing patients
            medical_history: "",
            user_id: "",
            passcode: "",
            caretakers: [], // Backend will add current user to this
            stage: "non_demented"
          }),
        });
  
        
        if (!response.ok) {
          const errorData = await response.json();
          let errorMessage = 'Registration failed';
        
          if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          } else if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map(d => d.msg).join(', ');
          } else if (typeof errorData.detail === 'object') {
            errorMessage = JSON.stringify(errorData.detail);
          }
        
          throw new Error(errorMessage);
        }
        await fetchPatientData();
      setActiveTab('dashboard');
      return;
    }

      // Frontend validation
      const errors = {};
      if (!formData.name?.trim()) errors.name = 'Name is required';
      if (!formData.age || formData.age < 1 || formData.age > 119) errors.age = 'Age must be 1-119';
      if (!['male', 'female', 'other'].includes(formData.gender?.toLowerCase())) {
        errors.gender = 'Gender must be male/female/other';
      }
      if (!/^[^@]+@[^@]+\.[^@]+$/.test(formData.contact_info.email)) {
        errors.email = 'Invalid email format';
      }
      if (!/^\d{10,15}$/.test(formData.contact_info.phone)) {
        errors.phone = 'Phone must be 10-15 digits';
      }
  
      if (Object.keys(errors).length) {
        throw new Error(Object.values(errors).join(', '));
      }
  
      const response = await fetch(`${API_BASE_URL}/api/register_patient`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          name: formData.name.trim(),
          age: Number(formData.age),
          gender: formData.gender.toLowerCase(),
          medical_history: formData.medical_history?.trim(),
          email: formData.contact_info.email.trim(),
          phone: formData.contact_info.phone.trim()
        }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Registration failed');
      }
  
      await fetchPatientData();
      setActiveTab('dashboard');
      
    } catch (error) {
      console.error('Registration error:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  

  const handleImageUpload = async (imageData) => {
    setIsLoading(true);
    
    try {
        // Validate input data structure
        if (!imageData || typeof imageData !== 'object') {
            throw new Error('Invalid image data format');
        }

        // Validate required fields
        if (!imageData.file) {
            throw new Error('No image file selected');
        }
        if (!imageData.patient_id) {
            throw new Error('Patient ID is required');
        }

        // Validate file properties
        if (!(imageData.file instanceof File)) {
            throw new Error('Invalid file format');
        }
        if (!imageData.file.type.startsWith('image/')) {
            throw new Error('Only image files are allowed');
        }
        if (imageData.file.size > 5 * 1024 * 1024) {
            throw new Error('Image size exceeds 5MB limit');
        }

        const formData = new FormData();
        formData.append('image', imageData.file);
        formData.append('description', imageData.description || 'No description provided');
        formData.append('patient_id', imageData.patient_id);

        const token = localStorage.getItem('token');
        if (!token) {
            throw new Error('Authentication token missing');
        }

        // Add timeout for the fetch request
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const response = await fetch(`${API_BASE_URL}/api/upload_image`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData,
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        // Handle HTTP errors
        if (!response.ok) {
            let errorDetail = `Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || JSON.stringify(errorData);
            } catch (e) {
                console.warn('Failed to parse error response');
            }
            throw new Error(`Upload failed: ${errorDetail}`);
        }

        const data = await response.json();
        
        // Verify the expected response structure
        if (!data.success) {
            throw new Error(data.message || 'Upload completed but reported failure');
        }

        // Refresh data
        await Promise.all([
            fetchNotifications(),
            fetchPatientData()
        ]);

        return {
            success: true,
            data: data,
            message: 'Image uploaded successfully'
        };

    } catch (error) {
        console.error('Image upload error:', error);
        return {
            success: false,
            error: error.message,
            message: error.message
        };
    } finally {
        setIsLoading(false);
    }
};

  const handleLogout = () => {
    localStorage.removeItem('token');
    window.location.href = '/'; 
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showProfileDropdown && !event.target.closest('.profile-container')) {
        setShowProfileDropdown(false);
      }
    };
  
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showProfileDropdown]);
 
  
  
  return (
      <div className="family-dashboard-container">
        {error && (
          <div className="family-alert-error">
            {error}
            <button className="dismiss-btn" onClick={() => setError(null)}>×</button>
          </div>
        )}
  
        <nav className="family-dashboard-nav">
          <div className="nav-left">
            <button 
              onClick={() => setActiveTab('dashboard')}
              className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            >
              <i className="fas fa-tachometer-alt"></i> Dashboard
            </button>
            <button 
              onClick={() => setActiveTab('register')}
              className={`nav-btn ${activeTab === 'register' ? 'active' : ''}`}
            >
              <i className="fas fa-user-plus"></i> {patientData ? 'Add Patient' : 'Register Patient'}
            </button>
          </div>
  
          <div className="nav-right">
            <div className="notifications-container">
              <button 
                className={`notifications-btn ${showNotifications ? 'active' : ''}`}
                onClick={() => setShowNotifications(!showNotifications)}
              >
                <i className="fas fa-bell"></i>
                {notifications.length > 0 && (
                  <span className="notification-badge">{notifications.length}</span>
                )}
              </button>
              
              {showNotifications && (
                <div className="notifications-dropdown">
                  <Notifications items={notifications} />
                </div>
              )}
            </div>
  
            <div className={`profile-container ${showProfileDropdown ? 'show-dropdown' : ''}`}>
              <button 
                className="profile-btn"
                onClick={() => setShowProfileDropdown(!showProfileDropdown)}
              >
                <i className="fas fa-user-circle"></i> {username}
              </button>
  
              {showProfileDropdown && (
                <div className="profile-dropdown">
                  <div className="profile-info">
                  <h3 className="usern">
                      User: <span className="un">{username}</span>
                    </h3>
                  </div>
                  <button 
                    className="logout-btn"
                    onClick={handleLogout}
                  >
                    <i className="fas fa-sign-out-alt"></i> Log Out
                  </button>
                </div>
              )}
            </div>
          </div>
        </nav>
  
        <div className="family-dashboard-content">
          {isLoading && (
            <div className="loading-overlay">
              <div className="loading-spinner"></div>
            </div>
          )}
  
          {activeTab === 'dashboard' && patientData && (
            <div className="dashboard-main">
              <PatientStats 
                stats={stats} 
                patient={patientData} 
                onMRIClick={() => setShowMRIModal(true)}
                onImageClick={() => setShowImageModal(true)}
                apiBaseUrl={API_BASE_URL}
              />
  
              {/* MRI Upload Modal */}
              {showMRIModal && (
                <div className="modal-overlay">
                  <div className="modal-content">
                    <div className="modal-header">
                      <h3>Upload MRI Scan for {patientData?.name || 'Patient'}</h3>
                      <button 
                        className="modal-close-btn"
                        onClick={() => setShowMRIModal(false)}
                      >
                        &times;
                      </button>
                    </div>
                    <MRIScanUpload 
                      apiBaseUrl={API_BASE_URL}
                      onUpload={async (result) => {
                        if (result.success) {
                          setShowMRIModal(false);
                          await fetchPatientData();
                        }
                      }}
                    />
                  </div>
                </div>
              )}
  
              {/* Image Upload Modal */}
              {showImageModal && (
                <div className="modal-overlay">
                  <div className="modal-content">
                    <div className="modal-header">
                      <h3>Upload Image for {patientData.name}</h3>
                      <button 
                        className="modal-close-btn"
                        onClick={() => setShowImageModal(false)}
                      >
                        &times;
                      </button>
                    </div>
                    <ImageUpload 
                      onUpload={handleImageUpload}
                      apiBaseUrl={API_BASE_URL}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
  
          {activeTab === 'register' && (
            <div className="registration-section">
              <div className="section-header">
                <h2>{patientData ? 'Add New Patient' : 'Patient Registration'}</h2>
                <p>Please provide the patient's details below</p>
              </div>
              
              <div className="registration-type-selector">
                <button 
                  onClick={() => setRegistrationType('new')}
                  className={`type-btn ${registrationType === 'new' ? 'active' : ''}`}
                >
                  <i className="fas fa-user-plus"></i> Register New Patient
                </button>
                <button 
                  onClick={() => setRegistrationType('existing')}
                  className={`type-btn ${registrationType === 'existing' ? 'active' : ''}`}
                >
                  <i className="fas fa-link"></i> Link Existing Patient
                </button>
              </div>
  
              {registrationType === 'existing' ? (
                <div className="link-patient-form">
                  <div className="form-group">
                    <label>Patient ID</label>
                    <input
                      type="text"
                      value={existingPatientId}
                      onChange={(e) => setExistingPatientId(e.target.value)}
                      placeholder="Enter patient ID"
                    />
                  </div>
                  <button 
                    onClick={() => handlePatientRegistration({})}
                    className="submit-btn"
                    disabled={!existingPatientId.trim()}
                  >
                    <i className="fas fa-link"></i> Link Patient
                  </button>
                </div>
              ) : (
                <PatientRegistrationForm 
                  onSubmit={handlePatientRegistration} 
                  isLoading={isLoading}
                />
              )}
            </div>
          )}
        </div>
      </div>
    );
  };
  
  export default FamilyHome;