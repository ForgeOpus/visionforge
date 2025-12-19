/**
 * Project Card Component
 * Displays individual project in dashboard grid with actions
 */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MoreVertical, Trash2, Clock } from 'lucide-react';
import { ProjectResponse, deleteProject } from '../lib/projectApi';

interface ProjectCardProps {
  project: ProjectResponse;
  onDeleted: () => void;
}

export const ProjectCard: React.FC<ProjectCardProps> = ({ project, onDeleted }) => {
  const navigate = useNavigate();
  const [showMenu, setShowMenu] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleCardClick = (e: React.MouseEvent) => {
    // Don't navigate if clicking on menu or delete button
    if ((e.target as HTMLElement).closest('.project-card-menu')) {
      return;
    }
    navigate(`/project/${project.id}`);
  };

  const handleDelete = async () => {
    try {
      setDeleting(true);
      await deleteProject(project.id);
      setShowDeleteConfirm(false);
      onDeleted();
    } catch (error) {
      console.error('Failed to delete project:', error);
      alert('Failed to delete project. Please try again.');
    } finally {
      setDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else if (diffDays < 30) {
      const weeks = Math.floor(diffDays / 7);
      return `${weeks} week${weeks > 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
      });
    }
  };

  const getFrameworkBadge = () => {
    if (project.framework === 'pytorch') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 bg-orange-100 text-orange-700 text-xs font-medium rounded-full border border-orange-200">
          🔥 PyTorch
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 bg-amber-100 text-amber-700 text-xs font-medium rounded-full border border-amber-200">
          ⚡ TensorFlow
        </span>
      );
    }
  };

  return (
    <>
      {/* Project Card */}
      <div
        onClick={handleCardClick}
        className="bg-card rounded-lg border border-border hover:border-primary hover:shadow-lg transition-all duration-200 cursor-pointer group relative overflow-hidden"
      >
        {/* Gradient accent on hover */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-purple-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-200"></div>

        <div className="p-5">
          {/* Header with title and menu */}
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-card-foreground truncate group-hover:text-primary transition-colors">
                {project.name}
              </h3>
            </div>

            {/* Menu Button */}
            <div className="project-card-menu relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(!showMenu);
                }}
                className="p-1 text-muted-foreground hover:text-foreground hover:bg-accent rounded transition-colors"
                aria-label="Project options"
              >
                <MoreVertical size={18} />
              </button>

              {/* Dropdown Menu */}
              {showMenu && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenu(false)}
                  ></div>
                  <div className="absolute right-0 mt-1 w-48 bg-popover rounded-lg shadow-lg border border-border py-1 z-20">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowMenu(false);
                        setShowDeleteConfirm(true);
                      }}
                      className="w-full px-4 py-2 text-left text-destructive hover:bg-destructive/10 flex items-center gap-2 transition-colors"
                    >
                      <Trash2 size={16} />
                      Delete Project
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Description */}
          <p className="text-muted-foreground text-sm mb-4 line-clamp-2 min-h-[2.5rem]">
            {project.description || 'No description provided'}
          </p>

          {/* Footer with framework and date */}
          <div className="flex items-center justify-between">
            {getFrameworkBadge()}

            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Clock size={14} />
              {formatDate(project.updated_at)}
            </div>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-card rounded-lg shadow-xl max-w-md w-full p-6">
            <h3 className="text-xl font-bold text-card-foreground mb-2">Delete Project?</h3>
            <p className="text-muted-foreground mb-6">
              Are you sure you want to delete <span className="font-semibold">{project.name}</span>?
              This action cannot be undone.
            </p>

            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                disabled={deleting}
                className="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {deleting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 size={16} />
                    Delete
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
