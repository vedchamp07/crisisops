"""
deployment/jira_adapter.py — Maps agent action outputs to Linear/Jira API calls.

Spec: "JIRA ADAPTER (jira_adapter.py)"

Uses identical field names and JSON schemas as the simulation.
Reads JIRA_API_KEY and JIRA_PROJECT_ID from environment variables.

For the demo:
    - Only submit_recovery_plan() makes a real API call (creates a new issue)
    - All other actions log what they would do without calling the API
    - dry_run=True prints the API payload without executing
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names for credentials
# ---------------------------------------------------------------------------
ENV_JIRA_API_KEY    = "JIRA_API_KEY"
ENV_JIRA_PROJECT_ID = "JIRA_PROJECT_ID"
ENV_JIRA_BASE_URL   = "JIRA_BASE_URL"   # e.g. "https://yourorg.atlassian.net"

# Default base URL for Linear (alternative to Jira)
DEFAULT_LINEAR_URL = "https://api.linear.app/graphql"

# Issue type for recovery plan issues
RECOVERY_PLAN_ISSUE_TYPE = "Task"
RECOVERY_PLAN_PRIORITY   = "High"


class JiraAdapter:
    """
    Translates CrisisOps agent actions to Linear/Jira API calls.

    All actions except submit_recovery_plan are logged only (no real API call).
    submit_recovery_plan creates a real Jira issue when dry_run=False.

    The JSON schemas for all actions are identical to the simulation so the
    same agent output can be used in both environments without modification.
    """

    def __init__(self, dry_run: bool = True) -> None:
        """
        Args:
            dry_run: If True, print API payloads without making any real calls.
                     Default True for safety.
        """
        self.dry_run = dry_run
        self._api_key    = os.environ.get(ENV_JIRA_API_KEY, "")
        self._project_id = os.environ.get(ENV_JIRA_PROJECT_ID, "")
        self._base_url   = os.environ.get(ENV_JIRA_BASE_URL, "")

        if not dry_run and (not self._api_key or not self._project_id):
            raise EnvironmentError(
                f"dry_run=False requires {ENV_JIRA_API_KEY} and "
                f"{ENV_JIRA_PROJECT_ID} environment variables to be set."
            )

    def dispatch(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch an agent action to the appropriate adapter method.

        Returns a result dict describing what was (or would have been) done.
        """
        action_type = action.get("action_type", "unknown")
        params = action.get("params", {})

        dispatch_table = {
            "query_status":             self._log_query,
            "query_member_report":      self._log_query,
            "query_observable_signals": self._log_query,
            "query_ticket":             self._log_query,
            "reassign_task":            self._handle_reassign_task,
            "communicate":              self._handle_communicate,
            "cut_scope":                self._handle_cut_scope,
            "escalate_risk":            self._handle_escalate_risk,
            "request_resource":         self._handle_request_resource,
            "update_timeline":          self._handle_update_timeline,
            "consult_expert":           self._log_query,
            "resolve_blocker":          self._handle_resolve_blocker,
            "submit_recovery_plan":     self._handle_submit_recovery_plan,
        }

        handler = dispatch_table.get(action_type, self._unknown_action)
        return handler(action_type, params)

    # ------------------------------------------------------------------
    # Log-only handlers (no API call)
    # ------------------------------------------------------------------

    def _log_query(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a read-only query action without making an API call."""
        logger.info("[JiraAdapter] %s %s — read-only, no API call", action_type, params)
        return {"logged": True, "action_type": action_type, "params": params}

    def _handle_reassign_task(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a task reassignment (would update issue assignee in Jira)."""
        payload = {
            "action": "update_issue_assignee",
            "issue_key": params.get("task_id"),
            "new_assignee": params.get("to_member_id"),
            "project_id": self._project_id,
        }
        return self._maybe_call("PUT", f"/issue/{params.get('task_id')}/assignee", payload)

    def _handle_communicate(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a stakeholder communication (would add a Jira comment)."""
        payload = {
            "action": "add_comment",
            "body": f"[{params.get('message_type', 'update')}] {params.get('content', '')}",
            "target": params.get("target", "both"),
            "project_id": self._project_id,
        }
        return self._maybe_call("POST", "/issue/PM-STATUS/comment", payload)

    def _handle_cut_scope(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a scope cut (would transition issue to Won't Do in Jira)."""
        payload = {
            "action": "transition_issue",
            "issue_key": params.get("task_id"),
            "transition": "Won't Do",
            "comment": params.get("justification", "Scope cut by PM"),
            "project_id": self._project_id,
        }
        return self._maybe_call("POST", f"/issue/{params.get('task_id')}/transitions", payload)

    def _handle_escalate_risk(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a risk escalation (would create a blocker issue in Jira)."""
        payload = {
            "action": "create_linked_issue",
            "issue_type": "Risk",
            "summary": f"Risk escalation: {params.get('crisis_id')}",
            "description": params.get("risk_description", ""),
            "priority": "High",
            "project_id": self._project_id,
        }
        return self._maybe_call("POST", "/issue", payload)

    def _handle_request_resource(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a resource request (would create a new Jira issue)."""
        payload = {
            "action": "create_issue",
            "issue_type": "Task",
            "summary": f"Resource request: {params.get('resource_type', 'headcount')}",
            "description": f"Requested for: {params.get('target_member_id', 'team')}",
            "project_id": self._project_id,
        }
        return self._maybe_call("POST", "/issue", payload)

    def _handle_update_timeline(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a timeline update (would update due dates in Jira)."""
        payload = {
            "action": "update_due_dates",
            "new_completion_date": params.get("new_completion_date", ""),
            "task_estimates": params.get("task_estimates", {}),
            "project_id": self._project_id,
        }
        return self._maybe_call("PUT", "/project/timeline", payload)

    def _handle_resolve_blocker(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a blocker resolution (would update issue status in Jira)."""
        payload = {
            "action": "transition_issue",
            "issue_key": params.get("task_id"),
            "transition": "In Progress",
            "comment": params.get("resolution_notes", "Blocker resolved by PM"),
            "project_id": self._project_id,
        }
        return self._maybe_call("POST", f"/issue/{params.get('task_id')}/transitions", payload)

    def _handle_submit_recovery_plan(
        self, action_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit recovery plan — the ONLY action that makes a real API call
        when dry_run=False.

        Creates a new Jira issue with the recovery plan as the description.
        """
        payload = {
            "fields": {
                "project":     {"key": self._project_id},
                "summary":     "Recovery Plan: " + params.get("plan_summary", "")[:60],
                "description": {
                    "type":    "doc",
                    "version": 1,
                    "content": [{
                        "type": "paragraph",
                        "content": [{"type": "text", "text": params.get("plan_summary", "")}],
                    }, {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Risk items: " + ", ".join(params.get("risk_items", []))}],
                    }, {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Timeline: " + params.get("timeline", "")}],
                    }],
                },
                "issuetype":  {"name": RECOVERY_PLAN_ISSUE_TYPE},
                "priority":   {"name": RECOVERY_PLAN_PRIORITY},
            }
        }

        if self.dry_run:
            print(f"[DRY RUN] POST /rest/api/3/issue")
            print(json.dumps(payload, indent=2))
            return {"dry_run": True, "would_create": payload}

        # Real API call — only for submit_recovery_plan
        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/rest/api/3/issue",
                json=payload,
                headers={
                    "Authorization": f"Basic {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info("[JiraAdapter] Created recovery plan issue: %s", data.get("key"))
            return {"success": True, "issue_key": data.get("key"), "url": data.get("self")}
        except Exception as exc:
            logger.error("[JiraAdapter] Failed to create issue: %s", exc)
            return {"success": False, "error": str(exc)}

    def _maybe_call(
        self, method: str, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        In dry_run mode, print the payload. Otherwise, log it (no real call for non-submit actions).
        """
        if self.dry_run:
            print(f"[DRY RUN] {method} {path}")
            print(json.dumps(payload, indent=2))
        else:
            logger.info("[JiraAdapter] Would call %s %s: %s", method, path, payload)
        return {"logged": True, "method": method, "path": path}

    def _unknown_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log a warning for unknown action types."""
        logger.warning("[JiraAdapter] Unknown action_type: %s", action_type)
        return {"logged": False, "error": f"Unknown action_type: {action_type}"}
