# Video Upload Frontend

A modern React + Vite frontend application for video upload and preview functionality.

## Features

- **Drag & Drop Video Upload**: Intuitive drag-and-drop interface for video files
- **Video Preview**: Real-time preview of selected videos before upload
- **Progress Tracking**: Visual progress indicators during upload
- **File Validation**: Automatic validation of video file formats
- **Responsive Design**: Mobile-friendly interface
- **Modern UI**: Clean, professional styling with smooth animations

## Tech Stack

- **React 19**: Latest React with modern hooks and features
- **Vite 6**: Fast build tool and development server
- **Axios**: HTTP client for API communication
- **Modern CSS**: Responsive design with CSS Grid and Flexbox

## Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Environment Configuration** (Optional):
   - Copy `.env.example` to `.env` if you need custom configuration
   - Update `VITE_BACKEND_URL` if your backend runs on a different port

3. **Development Server**:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:3000`

4. **Production Build**:
   ```bash
   npm run build
   ```

5. **Preview Production Build**:
   ```bash
   npm run preview
   ```

## Project Structure

```
frontend/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── index.js           # React entry point
│   └── index.css          # Global styles
├── home.jsx               # Main component
├── package.json           # Dependencies and scripts
├── vite.config.js         # Vite configuration
└── README.md             # This file
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Create production build
- `npm run preview` - Preview production build locally

## Configuration

### Environment Variables

The application supports the following environment variables:

- `VITE_BACKEND_URL` - Backend API URL (default: http://localhost:5000)
- `VITE_APP_TITLE` - Application title
- `VITE_MAX_FILE_SIZE` - Maximum file size for uploads

### Vite Configuration

The `vite.config.js` file includes:

- React plugin integration
- Development server configuration (port 3000)
- Build optimization settings
- Source map generation for debugging

## Browser Support

- Modern browsers with ES2015+ support
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Development Notes

- The application uses React 19 with modern hooks
- All components use functional components with hooks
- ESM modules are used throughout
- Hot module replacement is enabled in development

## Backend Integration

This frontend is designed to work with the Flask backend located in the `../backend` directory. Make sure the backend server is running on `http://localhost:5000` before using the upload functionality.

## Troubleshooting

1. **Port already in use**: Change the port in `vite.config.js`
2. **Backend connection issues**: Verify backend is running and CORS is configured
3. **Build errors**: Check Node.js version (16+ required)