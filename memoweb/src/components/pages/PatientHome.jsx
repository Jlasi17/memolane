import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaGamepad, FaUser, FaBell, FaHome, FaTrophy, FaChartLine, FaCalendarAlt, FaPills } from 'react-icons/fa';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import './PatientHome.css';


const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const PatientHome = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showGamesDropdown, setShowGamesDropdown] = useState(false);
  const [gameUser, setGameUser] = useState({
    level: 1,
    exp: 0,
    badges: [],
    games_played: {}
  });
  const [notifications, setNotifications] = useState([]);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [username, setUsername] = useState('');
  const [date, setDate] = useState(new Date());
  const [scheduleData, setScheduleData] = useState({
    appointments: [],
    medications: []
  });
  const [patientData, setPatientData] = useState(null);
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const [showLevelUp, setShowLevelUp] = useState(false);
  const [newLevel, setNewLevel] = useState(null);
  const [expAnimation, setExpAnimation] = useState(0);
  const [completedAppointments, setCompletedAppointments] = useState({});
  const [takenMedications, setTakenMedications] = useState({});



  useEffect(() => {
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
            throw new Error('No patient data found');
          }
          throw new Error(`Failed to fetch patient data: ${response.status}`);
        }

        const data = await response.json();
        setPatientData(data.patient);
        setStats(data.stats);
        return data.patient;
      } catch (error) {
        console.error('Error:', error);
        setError(error.message);
        return null;
      } finally {
        setIsLoading(false);
      }
    };

    // Fetch all required data
    const fetchAllData = async () => {
      const patientData = await fetchPatientData();
      if (!patientData) {
        console.error("No patient data available");
        return;
      }
      // Fetch game user data
try {
  const patientId = patientData?.patient_id || localStorage.getItem('patientId');
  console.log("Fetching game user for patient ID:", patientId); // Debug log
  
  if (patientId) {
    const response = await fetch(`${API_BASE_URL}/api/game_user/${patientId}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });

    console.log("Game user response status:", response.status); // Debug log

    if (!response.ok) {
      throw new Error(`Failed to fetch game user: ${response.status}`);
    }

    const data = await response.json();
    console.log("Game user data received:", data); // Debug log
    setGameUser(data);
  }
} catch (error) {
  console.error("Game user error:", error);
  // Maintain the existing default state
  setGameUser(prev => ({
    ...prev,
    level: 1,
    exp: 0,
    badges: [],
    games_played: {}
  }));
}

// Fetch schedule data
try {
  const patientId = patientData?.patient_id || localStorage.getItem('patientId');
  if (patientId) {
    const [appointmentsRes, medicationsRes] = await Promise.all([
      fetch(`${API_BASE_URL}/api/appointments?patient_id=${patientId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      }),
      fetch(`${API_BASE_URL}/api/medications?patient_id=${patientId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      })
    ]);

    const appointments = appointmentsRes.ok ? await appointmentsRes.json() : [];
    const medications = medicationsRes.ok ? await medicationsRes.json() : [];
    
    // Initialize completed appointments state
    const completedAppts = {};
    appointments.forEach(appt => {
      if (appt.completed) {
        completedAppts[appt.id] = true;
      }
    });
    setCompletedAppointments(completedAppts);
    
    // Initialize taken medications state
    const takenMeds = {};
    medications.forEach(med => {
      med.taken_times?.forEach(takenTime => {
        takenMeds[`${med.id}_${takenTime.time}`] = true;
      });
    });
    setTakenMedications(takenMeds);
    
    setScheduleData({
      appointments: Array.isArray(appointments) ? appointments : [],
      medications: Array.isArray(medications) ? medications : []
    });
  }
} catch (error) {
  console.error("Failed to fetch schedule data:", error);
  setScheduleData({
    appointments: [],
    medications: []
  });
}



      // Fetch user profile
      try {
        const response = await fetch(`${API_BASE_URL}/api/user`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setUsername(data.username);
      } catch (error) {
        console.error("Failed to fetch user profile:", error);
      }

      // Fetch schedule data
      try {
        const patientId = patientData?.patient_id || localStorage.getItem('patientId');
        if (patientId) {
          const [appointmentsRes, medicationsRes] = await Promise.all([
            fetch(`${API_BASE_URL}/api/appointments?patient_id=${patientId}`, {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              }
            }),
            fetch(`${API_BASE_URL}/api/medications?patient_id=${patientId}`, {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              }
            })
          ]);

          const appointments = await appointmentsRes.json();
          const medications = await medicationsRes.json();
          
          setScheduleData({
            appointments,
            medications
          });
        }
      } catch (error) {
        console.error("Failed to fetch schedule data:", error);
      }

      // Fetch notifications
      try {
        const response = await fetch(`${API_BASE_URL}/api/notifications`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setNotifications(data);
      } catch (error) {
        console.error("Failed to fetch notifications:", error);
      }
    };

    fetchAllData();
  }, []);
  const handleAppointmentComplete = async (appointmentId) => {
    try {
      // Optimistically update the UI
      setCompletedAppointments(prev => ({
        ...prev,
        [appointmentId]: true
      }));
  
      const response = await fetch(`${API_BASE_URL}/api/appointments/${appointmentId}/status`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ completed: true })
      });
  
      if (!response.ok) {
        // Revert if the request fails
        setCompletedAppointments(prev => {
          const newState = {...prev};
          delete newState[appointmentId];
          return newState;
        });
        throw new Error('Failed to update appointment status');
      }
    } catch (error) {
      console.error("Error updating appointment status:", error);
      setError(error.message);
    }
  };
  
  const handleMedicationTaken = async (medicationId, time) => {
    try {
      // Optimistically update the UI
      setTakenMedications(prev => ({
        ...prev,
        [`${medicationId}_${time}`]: true
      }));
  
      const response = await fetch(`${API_BASE_URL}/api/medications/${medicationId}/status`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ taken: true, time })
      });
  
      if (!response.ok) {
        // Revert if the request fails
        setTakenMedications(prev => {
          const newState = {...prev};
          delete newState[`${medicationId}_${time}`];
          return newState;
        });
        throw new Error('Failed to update medication status');
      }
    } catch (error) {
      console.error("Error updating medication status:", error);
      setError(error.message);
    }
  };

  const handleGameSelect = (game) => {
    setShowGamesDropdown(false);
    navigate(`/${game}`);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    localStorage.removeItem('patientId');
    navigate('/');
  };

  const expPercentage = gameUser && gameUser.level
  ? Math.min(100, (gameUser.exp / (gameUser.level * 100)) * 100)
  : 0;
  

  // Filter schedule items for the selected date
  const getScheduleForDate = (date) => {
    // Create a date-only string in local time (YYYY-MM-DD)
    const dateStr = new Date(date.getTime() - (date.getTimezoneOffset() * 60000))
      .toISOString()
      .split('T')[0];
    
    // Safely handle appointments
    const dailyAppointments = Array.isArray(scheduleData.appointments) 
      ? scheduleData.appointments.filter(appt => {
          // Compare with the stored date string directly
          return appt.date === dateStr;
        })
      : [];
  
    // Safely handle medications
    const dailyMedications = Array.isArray(scheduleData.medications) 
      ? scheduleData.medications.filter(med => {
          if (!med.created_at || !med.expires_at) return false;
          
          try {
            // Create date objects without timezone conversion
            const startDate = new Date(med.created_at.split('T')[0]);
            const endDate = new Date(med.expires_at.split('T')[0]);
            const currentDate = new Date(dateStr);
            
            return currentDate >= startDate && currentDate <= endDate;
          } catch (e) {
            console.error("Error parsing medication dates:", e);
            return false;
          }
        })
      : [];
  
    return {
      appointments: dailyAppointments,
      medications: dailyMedications
    };
  };

  const tileContent = ({ date, view }) => {
    if (view === 'month') {
      const dateStr = date.toISOString().split('T')[0];
      const hasAppointments = scheduleData.appointments?.some(
        appt => appt.date === dateStr
      );
      const hasMedications = scheduleData.medications?.some(med => {
        const startDate = new Date(med.created_at);
        const endDate = new Date(med.expires_at);
        return date >= startDate && date <= endDate;
      });
  
      return (
        <div className="calendar-indicators">
          {hasAppointments && <div className="appointment-indicator" />}
          {hasMedications && <div className="medication-indicator" />}
        </div>
      );
    }
  };

  const dailySchedule = getScheduleForDate(date);
  
  const getStageDescription = (stage) => {
    // Handle both string and numeric stage values
    const stageValue = String(stage); // Convert to string to handle both "1" and 1
    
    switch(stageValue) {
      case "0": return "No Dementia";
      case "1": return "Very Mild Dementia";
      case "2": return "Mild Dementia";
      case "3": return "Moderate Dementia";
      default: return "Stage not assessed";
    }
  };
  const LevelUpModal = () => {
    if (!showLevelUp) return null;
  
    return (
      <div className="level-up-modal">
        <div className="level-up-content">
          <div className="confetti">
            {[...Array(50)].map((_, i) => (
              <div key={i} className="confetti-piece" />
            ))}
          </div>
          <h2>Level Up! 🎉</h2>
          <h1>Level {newLevel}</h1>
          <button 
            className="close-button"
            onClick={() => setShowLevelUp(false)}
          >
            Awesome!
          </button>
        </div>
      </div>
    );
  };
  const saveScore = async (score, roundsCompleted) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/save-score`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          score: score,
          rounds_completed: roundsCompleted
        }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to save score');
      }
  
      const data = await response.json();
      
      // Handle level up
      if (data.leveled_up) {
        // Animate EXP bar
        const expGain = score * 10;
        animateExpBar(expGain);
        
        // Show level up modal after animation
        setTimeout(() => {
          setNewLevel(data.new_level);
          setShowLevelUp(true);
          
          // Refresh user data to get updated level and badges
          fetchGameUserData();
        }, 1000);
      }
      
      return data;
    } catch (error) {
      console.error("Error saving score:", error);
      throw error;
    }
  };
  
  const animateExpBar = (expGain) => {
    const duration = 1000; // 1 second
    const startTime = performance.now();
    
    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      setExpAnimation(progress * expGain);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  };
  
  const fetchGameUserData = async () => {
    try {
      const patientId = patientData?.patient_id || localStorage.getItem('patientId');
      if (patientId) {
        const response = await fetch(`${API_BASE_URL}/api/game_user/${patientId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setGameUser(data);
      }
    } catch (error) {
      console.error("Failed to fetch game user data:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading your data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error Loading Data</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Try Again</button>
      </div>
    );
  }
  return (
    <div className="patient-home-container">
      {/* Header */}
      <header className="patient-header">
        <div className="header-left">
          <h1>Memory Lane</h1>
          <p>Welcome back, {patientData?.name || 'Patient'}!</p>
        </div>
        
        <div className="header-right">
          <div 
            className="notification-icon" 
            onClick={() => setShowNotifications(!showNotifications)}
          >
            <FaBell />
            {notifications.length > 0 && (
              <span className="notification-badge">{notifications.length}</span>
            )}
          </div>
          
          <div className={`profile-container ${showProfileDropdown ? 'show-dropdown' : ''}`}>
            <button 
              className="profile-btn"
              onClick={() => setShowProfileDropdown(!showProfileDropdown)}
            >
              <FaUser /> {username}
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
                  <FaUser /> Log Out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="patient-nav">
        <button 
          className={`nav-button ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <FaHome /> Dashboard
        </button>
        
        <div className="games-dropdown">
          <button 
            className={`nav-button ${activeTab.startsWith('games') ? 'active' : ''}`}
            onClick={() => setShowGamesDropdown(!showGamesDropdown)}
          >
            <FaGamepad /> Games
          </button>
          
          {showGamesDropdown && (
            <div className="dropdown-content">
              <button onClick={() => handleGameSelect('cardgame')}>Memory Match</button>
              <button onClick={() => handleGameSelect('memotap')}>Memo Tap</button>
              <button onClick={() => handleGameSelect('puzzle')}>Puzzle Solver</button>
            </div>
          )}
        </div>

        <button 
          className={`nav-button ${activeTab === 'schedule' ? 'active' : ''}`}
          onClick={() => setActiveTab('schedule')}
        >
          <FaCalendarAlt /> Schedule
        </button>
      </nav>

      {/* Main Content */}
      <main className="patient-main">
        {activeTab === 'dashboard' && (
          <div className="dashboard-container">
            <div className="welcome-section">
            <h2>Your Cognitive Health Dashboard</h2>
            <p>
              Current Alzheimer's Stage: 
              <span className="info-value">
                {getStageDescription(patientData?.alzheimer_stage)}
              </span>
            </p>
          </div>
            
            {gameUser ? (
                <div className="progress-section">
                    <h3>Your Progress</h3>
                    
                    <div className="level-display">
                      <div className="level-badge">
                        <span>Level {gameUser.level}</span>
                      </div>
                      
                      <div className="exp-bar">
                        <div 
                          className="exp-progress" 
                          style={{ width: `${expPercentage}%` }}
                        ></div>
                        <span className="exp-text">
                          {gameUser.exp}/{gameUser.level * 100} EXP
                          {gameUser.exp < gameUser.level * 100 }
                        </span>
                      </div>
                    </div>
                
                {gameUser.badges && gameUser.badges.length > 0 ? (
                  <div className="badges-section">
                    <h4>Your Badges</h4>
                    <div className="badges-grid">
                      {gameUser.badges.map((badge, index) => (
                        <div key={index} className="badge-item">
                          <FaTrophy />
                          <span>{badge}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p>Play games to earn your first badge!</p>
                )}
              </div>
            ) : (
              <div className="progress-section loading">
                <p>Loading your progress data...</p>
              </div>
            )}
            
            <div className="quick-stats">
              <div className="stat-card">
                <FaChartLine />
                <h4>Last Scan</h4>
                <p>
                  {patientData?.alzheimer_stage === '0' ? 'Non Demented' : 
                   patientData?.alzheimer_stage === '1' ? 'Very Mild Demented' :
                   patientData?.alzheimer_stage === '2' ? 'Mild Demented' :
                   patientData?.alzheimer_stage === '3' ? 'Moderate Demented' : 
                   'No scan data'}
                </p>
                {patientData?.last_scan_date && (
                  <small>
                    {new Date(patientData.last_scan_date).toLocaleDateString()}
                  </small>
                )}
              </div>
              
              <div className="stat-card">
                <FaGamepad />
                <h4>Games Played</h4>
                <p>{gameUser?.games_played ? Object.values(gameUser.games_played).reduce((a, b) => a + (b || 0), 0) : 0}</p>
              </div>

              <div className="stat-card">
                <FaUser />
                <h4>Age</h4>
                <p>{patientData?.age || 'N/A'}</p>
              </div>

              <div className="stat-card">
                <FaPills />
                <h4>Medications</h4>
                <p>{patientData?.medications?.length || 0}</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'schedule' && (
          <div className="schedule-container">
            <h2>Your Schedule</h2>
            
            <div className="calendar-section">
              <Calendar
                onChange={setDate}
                value={date}
                className="react-calendar-custom"
              />
              
              <div className="schedule-items">
                <h3>Schedule for {date.toDateString()}</h3>
                
                {dailySchedule.appointments.length > 0 && (
                  <div className="appointments-list">
                    <h4><FaCalendarAlt /> Appointments</h4>
                    <ul>
                      {dailySchedule.appointments.map((appt) => {
                          const apptDate = new Date(appt.date);
                          const displayDate = apptDate.toLocaleDateString('en-US', {
                            day: 'numeric',
                            month: 'short',
                            year: 'numeric'
                          });
                          return(
                        <li key={appt.id} className={completedAppointments[appt.id] ? 'completed' : ''}>
                          <div className="appointment-info">
                            <strong>{appt.time}</strong> - with Dr. {appt.doctor_id || 'No description'}
                            {appt.doctor_name && <span> ({appt.doctor_name})</span>}
                            ,  <span className="appointment-date">{displayDate}</span>
                          </div>
                          {!completedAppointments[appt.id] && (
                            <button 
                              className="complete-btn"
                              onClick={() => handleAppointmentComplete(appt.id)}
                            >
                              Done
                            </button>
                          )}
                          {completedAppointments[appt.id] && (
                            <span className="completed-check">✓ Done</span>
                          )}
                        </li>
                          );
                        })}
                    </ul>
                  </div>
                )}
                
                {dailySchedule.medications.length > 0 && (
                    <div className="medications-list">
                      <h4><FaPills /> Medications</h4>
                      <ul>
                        {dailySchedule.medications.map((med) => (
                          <li key={med.id}>
                            <div className="medication-info">
                              <strong>{med.name}</strong>
                              <div className="medication-times">
                                {med.time.map((time) => (
                                  <div 
                                    key={`${med.id}_${time}`} 
                                    className={`time-slot ${takenMedications[`${med.id}_${time}`] ? 'taken' : ''}`}
                                  >
                                    {time}
                                    {!takenMedications[`${med.id}_${time}`] ? (
                                      <button
                                        className="take-btn"
                                        onClick={() => handleMedicationTaken(med.id, time)}
                                      >
                                        Taken
                                      </button>
                                    ) : (
                                      <span className="taken-check">✓</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                              {med.notes && <p className="med-notes">Notes: {med.notes}</p>}
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                
                {dailySchedule.appointments.length === 0 && dailySchedule.medications.length === 0 && (
                  <p>No scheduled items for this day</p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Notifications Panel */}
      {showNotifications && (
        <div className="notifications-panel">
          <div className="notifications-header">
            <h3>Notifications</h3>
            <button onClick={() => setShowNotifications(false)}>Close</button>
          </div>
          
          {notifications.length > 0 ? (
            <ul className="notifications-list">
              {notifications.map((notification, index) => (
                <li key={index} className={notification.read ? 'read' : 'unread'}>
                  <p>{notification.message}</p>
                  <small>{new Date(notification.created_at).toLocaleString()}</small>
                </li>
              ))}
            </ul>
          ) : (
            <p>No new notifications</p>
          )}
        </div>
      )}
    </div>
  );
};

export default PatientHome;