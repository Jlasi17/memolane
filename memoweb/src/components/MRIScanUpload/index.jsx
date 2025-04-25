import { useState, useRef, useEffect } from 'react';
import './styles.css';

const MRIScanUpload = ({ onUpload, apiBaseUrl }) => {
  const [formData, setFormData] = useState({
    notes: '',
    scan_file: null,
    patient_id: '',
    scan_date: new Date().toISOString().split('T')[0]
  });
  const [patients, setPatients] = useState([]);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loadingPatients, setLoadingPatients] = useState(true);
  const fileInputRef = useRef(null);

  // Fetch patients on component mount
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/user_patients`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch patients');
        }
        
        const data = await response.json();
        setPatients(data.patients || []);
        
        if (data.patients?.length > 0) {
          setFormData(prev => ({ ...prev, patient_id: "" }));
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingPatients(false);
      }
    };

    fetchPatients();
  }, [apiBaseUrl]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB limit");
      return;
    }
    setFormData(prev => ({ ...prev, scan_file: file }));
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsUploading(true);
    setError(null);
    setSuccess(null);
    setProcessingStatus("Uploading scan...");
    setPredictionResult(null);
    
    try {
      if (!formData.patient_id) throw new Error("Please select a patient");
      if (!formData.scan_file) throw new Error("Please select a scan file");
      if (!formData.scan_date) throw new Error("Please select a scan date");

      const data = new FormData();
      data.append('scan_file', formData.scan_file);
      data.append('patient_id', formData.patient_id);
      data.append('scan_date', formData.scan_date);
      if (formData.notes) data.append('notes', formData.notes);

      const token = localStorage.getItem('token');
      if (!token) throw new Error('No authentication token found');

      // Upload scan
      setProcessingStatus("Uploading MRI scan...");
      const uploadResponse = await fetch(`${apiBaseUrl}/api/upload_mri_scan`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: data
      });
      
      const uploadResult = await uploadResponse.json();
      
      if (!uploadResponse.ok || !uploadResult.success) {
        throw new Error(uploadResult.message || 'Upload failed');
      }
      
      setSuccess("Scan uploaded. Processing with AI model...");
      setProcessingStatus("Analyzing scan for Alzheimer's markers...");
      
      // Poll for results
      const scanId = uploadResult.scan_id;
      const finalResult = await pollForResults(scanId);
      
      setProcessingStatus("Analysis complete!");
      setPredictionResult(finalResult.alzheimer_prediction);
      onUpload(finalResult);
      
      // Reset form (keep patient_id)
      setFormData(prev => ({
        notes: '',
        scan_file: null,
        scan_date: new Date().toISOString().split('T')[0],
        patient_id: prev.patient_id
      }));
      
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.message);
      setProcessingStatus(null);
    } finally {
      setIsUploading(false);
    }
  };

  const pollForResults = async (scanId) => {
    const token = localStorage.getItem('token');
    let attempts = 0;
    const maxAttempts = 30; // 30 attempts with 2s delay = 1 minute timeout
    
    while (attempts < maxAttempts) {
      try {
        const response = await fetch(`${apiBaseUrl}/api/mri_scan/${scanId}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        const data = await response.json();
        
        if (data.processing_status === "completed") {
          return data;
        } else if (data.processing_status?.startsWith("failed")) {
          throw new Error(data.processing_status);
        }
        
        await new Promise(resolve => setTimeout(resolve, 2000)); // 2s delay
        attempts++;
        
      } catch (error) {
        throw new Error(`Failed to get results: ${error.message}`);
      }
    }
    
    throw new Error("Analysis timed out");
  };

  const getStageDescription = (stage) => {
    switch(stage) {
      case "0": return "No Dementia";
      case "1": return "Very Mild Dementia";
      case "2": return "Mild Dementia";
      case "3": return "Moderate Dementia";
      default: return "Unknown Stage";
    }
  };

  return (
    <div className="mri-scan-upload">
      <h3>Upload MRI Scan</h3>
      
      {/* Success Message */}
      {success && (
        <div className="alert-success">
          {success}
          <button onClick={() => setSuccess(null)}>×</button>
        </div>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="alert-error">
          {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}
      
      {/* Processing Status */}
      {processingStatus && (
        <div className="processing-status">
          <p>{processingStatus}</p>
          {processingStatus.includes("Analyzing") && (
            <div className="progress-spinner"></div>
          )}
        </div>
      )}
      
      {/* Prediction Results */}
      {predictionResult?.status === "completed" && (
        <div className="prediction-results">
          <h4>Analysis Results:</h4>
          <p><strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(2)}%</p>
          <p><strong>Stage:</strong> {getStageDescription(predictionResult.stage)}</p>
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Patient</label>
          {loadingPatients ? (
            <select disabled>
              <option>Loading patients...</option>
            </select>
          ) : (
            <select
              name="patient_id"
              value={formData.patient_id}
              onChange={handleChange}
              required
            >
              {patients.length === 0 ? (
                <option value="">No patients available</option>
              ) : (
                <>
                  <option value="">Select a patient</option>
                  {patients.map(patient => (
                    <option key={patient.patient_id} value={patient.patient_id}>
                      {patient.name} (ID: {patient.patient_id})
                    </option>
                  ))}
                </>
              )}
            </select>
          )}
        </div>
        
        <div className="form-group">
          <label>Scan Date</label>
          <input
            type="date"
            name="scan_date"
            value={formData.scan_date}
            onChange={handleChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label>Notes</label>
          <textarea
            name="notes"
            value={formData.notes}
            onChange={handleChange}
            rows="2"
          />
        </div>
        
        <div className="form-group">
          <label>MRI Scan File (DICOM/NIfTI/JPG/PNG, max 10MB)</label>
          <input
            type="file"
            accept=".dicom,.nii,.nii.gz,.jpg,.jpeg,.png"
            onChange={handleFileChange}
            ref={fileInputRef}
            required
          />
        </div>
        
        <button 
          type="submit" 
          className="upload-btn"
          disabled={isUploading || patients.length === 0}
        >
          {isUploading ? (
            <>
              <span className="spinner"></span>
              Uploading...
            </>
          ) : 'Upload MRI Scan'}
        </button>
      </form>
    </div>
  );
};

export default MRIScanUpload;