import { useState, useEffect } from 'react';
import {
  PatientRegistrationForm,
  MRIScanUpload,
  ScheduleAppointment,
  AddMedication,
  PatientStatsDoc,
  Notifications
} from '../index';
import './docstyles.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const DoctorHome = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [patientData, setPatientData] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [registrationType, setRegistrationType] = useState('new');
  const [existingPatientId, setExistingPatientId] = useState('');
  const [showMRIModal, setShowMRIModal] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [showMedicationModal, setShowMedicationModal] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [username, setUsername] = useState('');
  const [showNotifications, setShowNotifications] = useState(false);
  const [appointments, setAppointments] = useState([]);
  const [medications, setMedications] = useState([]);
  const [patients, setPatients] = useState([]);

  // Fetch all necessary data on component mount
  useEffect(() => {
    const initializeData = async () => {
      try {
        await Promise.all([
          fetchPatientData(),
          fetchNotifications(),
          fetchUsername(),
          fetchAppointments(),
          fetchMedications(),
          fetchPatients()
        ]);
      } catch (error) {
        console.error('Initialization error:', error);
      }
    };
    initializeData();
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

  const fetchPatients = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/user_patients`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch patients');
      }
      
      const data = await response.json();
      setPatients(data.patients || []);
    } catch (err) {
      console.error('Error fetching patients:', err);
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
        throw new Error('User info not available');
      }
      
      const data = await response.json();
      setUsername(data.username || data.email || 'User');
    } catch (error) {
      console.error('Error fetching username:', error);
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

  const fetchAppointments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/appointments`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      const sortedAppointments = data.sort((a, b) => {
        const dateA = new Date(`${a.date}T${a.time}`);
        const dateB = new Date(`${b.date}T${b.time}`);
        return dateA - dateB;
      });
      setAppointments(sortedAppointments);
    } catch (error) {
      console.error('Error fetching appointments:', error);
    }
  };

  const fetchMedications = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/medications`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setMedications(data);
    } catch (error) {
      console.error('Error fetching medications:', error);
    }
  };

  const handlePatientRegistration = async (formData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      if (registrationType === 'existing') {
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
            name: "Linked Patient",
            age: 65,
            gender: "other",
            email: `replay.test@gmail.com`,
            phone: "0000000000",
            medical_history: "",
            user_id: "",
            passcode: "",
            caretakers: [],
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

  const handleAddAppointment = async (appointmentData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Verify we have a token
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found');
      }
  
      // Verify API base URL
      if (!API_BASE_URL) {
        throw new Error('API base URL not configured');
      }
  
      const response = await fetch(`${API_BASE_URL}/api/appointments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(appointmentData)
      });
  
      // Handle HTTP errors
      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        throw new Error(errorData.detail || 'Failed to add appointment');
      }
  
      const data = await response.json();
      await fetchAppointments();
      setShowScheduleModal(false);
      return data;
    } catch (error) {
      console.error('Error adding appointment:', error);
      setError(error.message);
      throw error; // Re-throw to handle in the calling component
    } finally {
      setIsLoading(false);
    }
  };

  
const handleCompleteAppointment = async (appointmentId) => {
  try {
    if (!appointmentId) {
      throw new Error("No appointment ID provided");
    }
    
    const response = await fetch(`${API_BASE_URL}/api/appointments/${appointmentId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to complete appointment');
    }

    await fetchAppointments();
  } catch (error) {
    console.error('Error completing appointment:', error);
    setError(error.message);
  }
};


  const handleAddMedication = async (medicationData) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/medications`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(medicationData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add medication');
      }

      await fetchMedications();
      setShowMedicationModal(false);
    } catch (error) {
      console.error('Error adding medication:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteMedication = async (medicationId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/medications/${medicationId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to delete medication');
      }

      await fetchMedications();
    } catch (error) {
      console.error('Error deleting medication:', error);
      setError(error.message);
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
      if (showNotifications && !event.target.closest('.notifications-container')) {
        setShowNotifications(false);
      }
    };
  
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showProfileDropdown, showNotifications]);

  return (
    <div className="dashboard-container">
      {error && (
        <div className="alert-error">
          {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <nav className="dashboard-nav">
        <div className="nav-left">
          <button 
            onClick={() => setActiveTab('dashboard')}
            className={activeTab === 'dashboard' ? 'active' : ''}
          >
            <i className="fas fa-tachometer-alt"></i> Dashboard
          </button>
          <button 
            onClick={() => setActiveTab('register')}
            className={activeTab === 'register' ? 'active' : ''}
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
              <i className="fas fa-user-circle"></i>  {username}
            </button>

            {showProfileDropdown && (
              <div className="profile-dropdown">
                <div className="profile-info">
                <h3 className="usern">
                      User: Dr. <span className="un">{username}</span>
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

      <div className="dashboard-content">
        {isLoading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Loading...</p>
          </div>
        )}

        {activeTab === 'dashboard' && patientData && (
          <div className="dashboard-main">
            <PatientStatsDoc 
              stats={stats} 
              patient={patientData} 
              onMRIClick={() => setShowMRIModal(true)}
              onScheduleClick={() => setShowScheduleModal(true)}
              onMedicationClick={() => setShowMedicationModal(true)}
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
                    patientId={patientData.patient_id}
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

            {/* Schedule Appointment Modal */}
            {showScheduleModal && (
              <div className="modal-overlay">
                <div className="modal-content">
                  <div className="modal-header">
                    <h3>Manage Appointments</h3>
                    <button 
                      className="modal-close-btn"
                      onClick={() => setShowScheduleModal(false)}
                    >
                      &times;
                    </button>
                  </div>
                  <ScheduleAppointment 
                    appointments={appointments}
                    onAddAppointment={handleAddAppointment}
                    onCompleteAppointment={handleCompleteAppointment}
                    apiBaseUrl={API_BASE_URL}
                  />
                </div>
              </div>
            )}

            {/* Add Medication Modal */}
            {showMedicationModal && (
              <div className="modal-overlay">
                <div className="modal-content">
                  <div className="modal-header">
                    <h3>Manage Medications</h3>
                    <button 
                      className="modal-close-btn"
                      onClick={() => setShowMedicationModal(false)}
                    >
                      &times;
                    </button>
                  </div>
                  <AddMedication 
                    medications={medications}
                    setMedications={setMedications}
                    onAddMedication={handleAddMedication}
                    onDeleteMedication={handleDeleteMedication}
                    patientData={patientData}
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
                className={registrationType === 'new' ? 'active' : ''}
              >
                <i className="fas fa-user-plus"></i> Register New Patient
              </button>
              <button 
                onClick={() => setRegistrationType('existing')}
                className={registrationType === 'existing' ? 'active' : ''}
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

export default DoctorHome;